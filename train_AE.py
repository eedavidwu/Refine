import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import Models.AE_Models.Auto_Encoder as AE_Models
import Models.AE_Models.Attention_AE as Atten_AE

import os
import argparse
import cv2
from random import randint
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def train(args,auto_encoder,trainloader,testloader,train_snr):

    #model_name:
    model_name=args.model
    # Define an optimizer and criterion
    criterion = nn.MSELoss()
    optimizer = optim.Adam(auto_encoder.parameters())
    
    #Start Train:
    batch_iter=(trainloader.dataset.__len__() // trainloader.batch_size)
    print_iter=int(batch_iter/2)
    best_psnr=0
    epoch_last=0
    best_loss=1

    #whether resume:
    if args.resume==True:
        model_path='./checkpoints/ADJSCC' 
        model_path=os.path.join(model_path,'AE_SNR_'+str(train_snr)+'.pth')
        checkpoint=torch.load(model_path)
        epoch_last=checkpoint["epoch"]
        auto_encoder.load_state_dict(checkpoint["net"])

        #optimizer=optimizer.load_state_dict(checkpoint["op"])
        best_psnr=checkpoint["Best_PSNR"]

        print("Load model:",model_path)
        print("Model is trained in SNR: ",train_snr," with PSNR:",best_psnr," at epoch ",epoch_last)
        auto_encoder = auto_encoder.cuda()

    for epoch in range(args.all_epoch):
        auto_encoder.train()
        running_loss = 0.0
        step=0
        #print('Epoch ',str(epoch),' trained with SNR: ',channel_flag)
        
        for batch_idx, (inputs, _) in enumerate(trainloader, 0):
            inputs = Variable(inputs.cuda())
            # set a random noisy:            
            # ============ Forward ============
            encoded_out,outputs = auto_encoder(inputs,train_snr)
            # ============ Backward ============
            optimizer.zero_grad()
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # ============ Ave_loss compute ============
            step=step+1
            running_loss += loss.data

        print('Epoch:[',epoch,']',", loss : " ,float(running_loss/step))

        if (epoch %2==0):
            if model_name=='DLJSCC':
                ##Validate:
                validate_snr=train_snr
                val_ave_psnr=compute_AvePSNR(auto_encoder,testloader,validate_snr)
                if val_ave_psnr > best_psnr:
                    best_psnr=val_ave_psnr
                    print('Find one best model with best PSNR:',best_psnr,' under SNR: ',train_snr)
                    checkpoint={
                        "model_name":args.model,
                        "net":auto_encoder.state_dict(),
                        "op":optimizer.state_dict(),
                        "epoch":epoch,
                        "SNR":train_snr,
                        "Best_PSNR":best_psnr
                    }
                    PSNR_list=[]
                    for i in [1,5,10,15,19]:
                    #for i in [1]:
                        ave_PSNR_test=compute_AvePSNR(auto_encoder,testloader,i)
                        PSNR_list.append(ave_PSNR_test)
                    print(PSNR_list)
                    SNR_path='./checkpoints/SNR_'+str(train_snr)  
                    check_dir(SNR_path)      
                    save_path=os.path.join(SNR_path,'AE_SNR_'+str(train_snr)+'.pth')
                    torch.save(checkpoint, save_path)
                    print('Saving Model at epoch',epoch)
            elif model_name=='ADJSCC':
                ##Validate:
                PSNR_list=[]
                for i in [1,5,10,15,19]:
                    validate_snr=i
                    val_ave_psnr=compute_AvePSNR(auto_encoder,testloader,validate_snr)
                    PSNR_list.append(val_ave_psnr)
                ave_PSNR=np.mean(PSNR_list)
                if ave_PSNR > best_psnr:
                    best_psnr=ave_PSNR
                    print('Find one best model with best PSNR:',best_psnr,' under SNR: ',train_snr)
                    checkpoint={
                        "model_name":args.model,
                        "net":auto_encoder.state_dict(),
                        "op":optimizer.state_dict(),
                        "epoch":epoch,
                        "SNR":train_snr,
                        "Best_PSNR":best_psnr
                    }
                    print(PSNR_list)
                    SNR_path='./checkpoints/ADJSCC' 
                    check_dir(SNR_path)      
                    save_path=os.path.join(SNR_path,'AE_SNR_'+str(train_snr)+'.pth')
                    torch.save(checkpoint, save_path)
                    print('Saving Model at epoch',epoch)
                 
def compute_AvePSNR(model,dataloader,snr):
    psnr_all_list = []
    model.eval()
    MSE_compute = nn.MSELoss(reduction='none')
    for batch_idx, (inputs, _) in enumerate(dataloader, 0):
        b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
        inputs = Variable(inputs.cuda())
        encoded_out,outputs = model(inputs,snr)
        MSE_each_image = (torch.sum(MSE_compute(outputs, inputs).view(b,-1),dim=1))/(c*h*w)
        PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
        one_batch_PSNR=PSNR_each_image.data.cpu().numpy()
        psnr_all_list.extend(one_batch_PSNR)
    Ave_PSNR=np.mean(psnr_all_list)
    return Ave_PSNR


def main():
    parser = argparse.ArgumentParser()
    #Train:
    parser.add_argument("--best_ckpt_path", default='./ckpts/', type=str,help='best model path')
    parser.add_argument("--all_epoch", default='150', type=int,help='Train_epoch')
    parser.add_argument("--best_choice", default='loss', type=str,help='select epoch [loss/PSNR]')
    parser.add_argument("--flag", default='train', type=str,help='train or eval for JSCC')

    # Model and Channel:
    parser.add_argument("--model", default='DLJSCC', type=str,help='Model select: DLJSCC/ADJSCC')
    parser.add_argument("--tcn", default=8, type=int,help='tansmit_channel_num for djscc')
    parser.add_argument("--channel_type", default='awgn', type=str,help='awgn/slow fading/burst')
    parser.add_argument("--snr", default=10,type=int,help='awgn/slow fading/burst')

    #parser.add_argument("--const_snr", default=True,help='SNR (db)')
    #parser.add_argument("--input_const_snr", default=1, type=float,help='SNR (db)')
    parser.add_argument("--input_snr_max", default=20, type=float,help='SNR (db)')
    parser.add_argument("--input_snr_min", default=0, type=int,help='SNR (db)')
    parser.add_argument("--resume", default=False,type=bool, help='Load past model')

    GPU_ids = [0,1,2,3]



    global args
    args=parser.parse_args()
    if args.model=='DLJSCC':
        auto_encoder=AE_Models.Classic_JSCC(args)
        train_snr=args.snr

    elif args.model=='ADJSCC':
        auto_encoder=Atten_AE.Attention_JSCC(args)
        train_snr='random'

    auto_encoder = nn.DataParallel(auto_encoder,device_ids = GPU_ids)
    auto_encoder = auto_encoder.cuda()
    print("Create the model:",args.model)

    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2)

    #for i in [1,4,7,10,13,19]:
    print("############## Train model with SNR: ",train_snr," ##############")
    train(args,auto_encoder,trainloader,testloader,train_snr)

if __name__ == '__main__':
    main()
