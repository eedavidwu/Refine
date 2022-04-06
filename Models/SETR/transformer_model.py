import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
from torch.autograd import Variable


def swish(x):
    return x * torch.sigmoid(x)

def gelu(x):
    """ 
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "mish": mish}

class TransConfig(object):
    
    def __init__(
        self,
        patch_size,
        in_channels,
        out_channels,
        sample_rate=4,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=6,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    ):  
        self.sample_rate = sample_rate
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class TransLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(TransLayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
       

    def forward(self, x):

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
      
class TransEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        input_shape = input_ids.size()
    
        seq_length = input_shape[1]
        device = input_ids.device
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape[:2])
       
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_ids + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransSelfAttention(nn.Module):
    def __init__(self, config: TransConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        ## 最后xshape (batch_size, num_attention_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 注意力加权
        context_layer = torch.matmul(attention_probs, value_layer)
        # 把加权后的V reshape, 得到[batch_size, length, embedding_dimension]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class TransSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = TransSelfAttention(config)
        self.output = TransSelfOutput(config)

    def forward(
        self,
        hidden_states,
    ):
        self_outputs = self.self(hidden_states)
        attention_output = self.output(self_outputs, hidden_states)
        
        return attention_output


class TransIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] ## relu 

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TransOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TransAttention(config)
        self.intermediate = TransIntermediate(config)
        self.output = TransOutput(config)

    def forward(
        self,
        hidden_states
    ):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class Decoder_Dense2d(nn.Module):
    def __init__(self, config,tcn):
        super(Decoder_Dense2d, self).__init__()
        self.dense = nn.Linear(tcn, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class Siam_linear(nn.Module):
    def __init__(self, config,inshape,out_shape):
        super(Siam_linear, self).__init__()
        self.dense_1 = nn.Linear(inshape, config.hidden_size)
        self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_cat = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.out_shape=out_shape
        if self.out_shape!=0:
            self.dense_last = nn.Linear(config.hidden_size, out_shape)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input):
        #brunch 0:
        hidden_states_0_1 = self.dense_3(self.transform_act_fn(self.dense_2(self.transform_act_fn(self.dense_1(input*1)))))
        #brunch 1:
        hidden_states_1_1 = self.dense_3(self.transform_act_fn(self.dense_2(self.transform_act_fn(self.dense_1(input*(-1))))))

        aggregate_state=torch.cat((hidden_states_0_1,hidden_states_1_1),dim=2)
        hidden_state_out= self.LayerNorm(self.transform_act_fn(self.dense_cat(aggregate_state)))
        return hidden_state_out

class Att_TransEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([TransLayer(config) for _ in range(config.num_hidden_layers)])
        self.AL = nn.ModuleList([AL_Module(config.hidden_size) for _ in range((config.num_hidden_layers)//2)])

    def forward(
        self,
        hidden_states,
        snr,
        output_all_encoded_layers=True,
    ):
        all_encoder_layers = []
        
        for i, layer_module in enumerate(self.layer):
            layer_output = layer_module(hidden_states)
            if ((i+1)%2==0):
                y_in=self.AL[(i-1)//2](layer_output,snr)
                layer_output=y_in+layer_output
            hidden_states = layer_output
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers

class TransEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([TransLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        output_all_encoded_layers=True,
    ):
        all_encoder_layers = []
        
        for i, layer_module in enumerate(self.layer):
            layer_output = layer_module(hidden_states)
            hidden_states = layer_output
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers
class ReciverModel2d(nn.Module):
    def __init__(self, config,input_length):
        super(ReciverModel2d, self).__init__()
        self.config = config
        self.dense = Decoder_Dense2d(config,input_length)
        #self.dense=Siam_linear(config,input_length,0)
        self.embeddings = TransEmbeddings(config)
        self.encoder = TransEncoder(config)

    def forward(
        self,
        input_ids,
        output_all_encoded_layers=True,
       
    ):  
        dense_out = self.dense(input_ids)
        embedding_output = self.embeddings(
            input_ids=dense_out
        )
        encoder_layers = self.encoder(
            embedding_output,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        #sequence_output = encoder_layers[-1]
        
        if not output_all_encoded_layers:
            # 如果不用输出所有encoder层
            encoder_layers = encoder_layers[-1]
        return encoder_layers

class InputDense2d(nn.Module):
    def __init__(self, config,tcn):
        super(InputDense2d, self).__init__()
        self.dense = nn.Linear(config.patch_size[0] * config.patch_size[1] * config.in_channels+48, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class InputDense3d(nn.Module):
    def __init__(self, config):
        super(InputDense3d, self).__init__()
        self.dense = nn.Linear(config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * config.in_channels, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class Input_denoise_Dense2d(nn.Module):
    def __init__(self, config,tcn):
        super(Input_denoise_Dense2d, self).__init__()
        self.dense_1 = nn.Linear(config.patch_size[0] * config.patch_size[1] * config.in_channels+tcn, config.hidden_size)
        #self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
        #self.dense_3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_last = nn.Linear(config.hidden_size*2, config.hidden_size)

        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        #brunch 0:
        hidden_states_0_1 = self.LayerNorm(self.transform_act_fn(self.dense_1(hidden_states*1)))
        #hidden_states_0_2 = self.transform_act_fn(self.dense_2(hidden_states_0_1))
        #hidden_states_0_3 = self.dense_3(hidden_states_0_2)
        #brunch 1:
        hidden_states_1_1 = self.LayerNorm(self.transform_act_fn(self.dense_1(hidden_states*(-1))))
        #hidden_states_1_2 = self.transform_act_fn(self.dense_2(hidden_states_1_1))
        #hidden_states_1_3 = self.dense_3(hidden_states_1_2)

        hidden_state_out=self.dense_last(torch.cat((hidden_states_0_1,hidden_states_1_1),dim=2))
        hidden_state_out = self.LayerNorm(self.transform_act_fn(hidden_state_out))
        return hidden_state_out

class TransModel2d(nn.Module):
    def __init__(self, config,tcn):
        super(TransModel2d, self).__init__()
        self.config = config
        self.dense = InputDense2d(config,tcn)
        #self.dense = Siam_linear(config,config.patch_size[0] * config.patch_size[1] * config.in_channels+tcn,0)
        self.embeddings = TransEmbeddings(config)
        self.encoder = TransEncoder(config)

    def forward(
        self,
        input_ids,
        output_all_encoded_layers=True,
       
    ):  
        #original:
        dense_out = self.dense(input_ids)
        #dense_out=self.denoise_dense(input_ids)
        #new design:

        embedding_output = self.embeddings(
            input_ids=dense_out
        )
        encoder_layers = self.encoder(
            embedding_output,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        #sequence_output = encoder_layers[-1]
        
        if not output_all_encoded_layers:
            # 如果不用输出所有encoder层
            encoder_layers = encoder_layers[-1]
        return encoder_layers

class AL_Module(nn.Module):
    def __init__(self,fc_in):
        super(AL_Module, self).__init__()
        #self.Ave_Pooling = nn.AdaptiveAvgPool2d(1)
        self.FC_1 = nn.Linear(fc_in+1,fc_in//16)
        self.FC_2 = nn.Linear(fc_in//16,fc_in)

    def forward(self, inputs,snr):
        out_pooling=torch.mean(inputs,dim=1)
        b=inputs.shape[0]
        c=inputs.shape[2]
        in_snr=Variable(torch.tensor(snr,dtype=torch.float).cuda()).view(b,-1)
        #in_snr=torch.tensor(snr,dtype=torch.float).cuda().repeat(b).view(b,1)
        in_fc=torch.cat((in_snr,out_pooling),dim=1)
        out_fc_1=self.FC_1(in_fc)
        out_fc_1_relu=torch.nn.functional.relu(out_fc_1)
        out_fc_2=self.FC_2(out_fc_1_relu)
        out_fc_2_sig=torch.sigmoid(out_fc_2).view(b,1,c)
        out=out_fc_2_sig*inputs
        return out
class Att_TransModel2d(nn.Module):
    def __init__(self, config,tcn):
        super(Att_TransModel2d, self).__init__()
        self.config = config
        hidden_layer_num=config.hidden_size
        self.dense = InputDense2d(config,tcn)
        self.embeddings = TransEmbeddings(config)
        self.encoder = Att_TransEncoder(config)

    def forward(
        self,
        input_ids,
        snr,
        output_all_encoded_layers=True,
       
    ):  
        dense_out = self.dense(input_ids)
        embedding_output = self.embeddings(
            input_ids=dense_out
        )
        #sequence_len=embedding_output.shape[1]
        #snr=torch.from_numpy(snr.reshape(-1,1,1)).repeat(1,sequence_len,1).cuda()
        encoder_layers = self.encoder(
            embedding_output,
            snr,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        
        if not output_all_encoded_layers:
            # 如果不用输出所有encoder层
            encoder_layers = encoder_layers[-1]
        return encoder_layers


class TransModel3d(nn.Module):

    def __init__(self, config):
        super(TransModel3d, self).__init__()
        self.config = config
        self.dense = InputDense3d(config)
        self.embeddings = TransEmbeddings(config)
        self.encoder = TransEncoder(config)

    def forward(
        self,
        input_ids,
        output_all_encoded_layers=True,
       
    ):  
        dense_out = self.dense(input_ids)
        embedding_output = self.embeddings(
            input_ids=dense_out
        )
        encoder_layers = self.encoder(
            embedding_output,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        sequence_output = encoder_layers[-1]
        
        if not output_all_encoded_layers:
            # 如果不用输出所有encoder层
            encoder_layers = encoder_layers[-1]
        return encoder_layers