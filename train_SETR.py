import os
#os.environ["CUDA_VISIBLE_DEVICES"] ="0"
#GPU_ids = [0]
GPU_ids = [0,1,2,3]
import torch 
from Models.SETR.transformer_seg import SETRModel
import torchvision
import torch
import torch.nn as nn 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import numpy as np
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device is " + str(device))
def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def compute_AvePSNR(model,dataloader,snr):
    psnr_all_list = []
    model.eval()
    MSE_compute = nn.MSELoss(reduction='none')
    for batch_idx, (inputs, _) in enumerate(dataloader, 0):
        b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
        inputs = inputs.cuda()
        outputs= model(inputs,snr)
    
        MSE_each_image = (torch.sum(MSE_compute(outputs, inputs).view(b,-1),dim=1))/(c*h*w)
        PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
        one_batch_PSNR=PSNR_each_image.data.cpu().numpy()
        psnr_all_list.extend(one_batch_PSNR)
    Ave_PSNR=np.mean(psnr_all_list)
    return Ave_PSNR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Train:
    parser.add_argument("--best_ckpt_path", default='./ckpts/', type=str,help='best model path')
    parser.add_argument("--all_epoch", default='2000', type=int,help='Train_epoch')
    parser.add_argument("--best_choice", default='loss', type=str,help='select epoch [loss/PSNR]')
    parser.add_argument("--flag", default='train', type=str,help='train or eval for JSCC')

    # Model and Channel:
    parser.add_argument("--model", default='SETR', type=str,help='Model select: SETR/ADSETR/')
    #parser.add_argument("--tcn", default=6, type=int,help='tansmit_channel_num for djscc')
    #parser.add_argument("--channel_type", default='awgn', type=str,help='awgn/slow fading/burst')
    parser.add_argument("--snr", default=4, type=int,help='awgn/slow fading/')

    #parser.add_argument("--const_snr", default=True,help='SNR (db)')
    #parser.add_argument("--input_const_snr", default=1, type=float,help='SNR (db)')
    parser.add_argument("--input_snr_max", default=20, type=float,help='SNR (db)')
    parser.add_argument("--input_snr_min", default=0, type=int,help='SNR (db)')
    parser.add_argument("--resume", default=False,type=bool, help='Load past model')
    args=parser.parse_args()
    check_dir('checkpoints')
    check_dir('data')

    if args.model=='SETR':
        #64*6 1/8 ->(4,4) tcn=6/iter
        #print("64*8 1/6 ->(4,4) tcn=8/2")
        print("64*16 1/3 -> tcn=16/4")

        print('head: 8')
        iter_num=4
        tcn=16//iter_num
        print('iter: 2')
        #print("16*24 1/8->(8,8) tcn=24")
        
        model = SETRModel(patch_size=(4, 4), 
                        in_channels=3, 
                        out_channels=3, 
                        hidden_size=256, 
                        num_hidden_layers=4, 
                        num_attention_heads=4, 
                        intermediate_size=1024,
                        tcn=tcn,iteration=4)
        channel_snr=args.snr
        #channel_snr='random'

    print('snr:',channel_snr)
    print(model)
    print("############## Train model",args.model,",with SNR: ",channel_snr," ##############")
    if len(GPU_ids)>1:
        model = nn.DataParallel(model,device_ids = GPU_ids)
    model = model.cuda()
    #print(model)

    #model.to(device)
    #transform = transforms.Compose([transforms.ToTensor(),
    #                           transforms.Normalize(mean=[0.5],std=[0.5])])
    #
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)


    optimizer = torch.optim.Adam(model.parameters(),lr=0.00005)
    #optimizer = torch.optim.Adam(model.parameters())

    loss_func = nn.MSELoss()
    best_psnr=0
    epoch_last=0

    #for in_data, label in tqdm(data_loader_train, total=len(data_loader_train)):
    if args.resume==True:
        model_path='./checkpoints/SNR_T_10/slim_SETR_double_2_iter_feed_all_SNR_10.pth'
        checkpoint=torch.load(model_path)
        epoch_last=checkpoint["epoch"]
        model.load_state_dict(checkpoint["net"])

        optimizer.load_state_dict(checkpoint["op"])
        best_psnr=checkpoint["Best_PSNR"]
        Trained_SNR=checkpoint['SNR']

        print("Load model:",model_path)
        print("Model is trained in SNR: ",Trained_SNR," with PSNR:",best_psnr," at epoch ",epoch_last)

    for epoch in range(epoch_last,args.all_epoch):
        step = 0
        report_loss = 0
        #val_ave_psnr=compute_AvePSNR(model,testloader,1)

        for in_data, label in trainloader:
            batch_size = len(in_data)
            in_data = in_data.to(device) 
            #label = label.to(device)
            optimizer.zero_grad()
            step += 1
            out= model(in_data,channel_snr)
            loss = loss_func(out, in_data)
            loss.backward()
            optimizer.step()
            report_loss += loss.item()
        print('Epoch:[',epoch,']',", loss : " ,str(report_loss/step))

        if (((epoch % 4 == 0) and (epoch>50)) or (epoch==0)):
            if args.model=='SETR':
                if channel_snr=='random':
                    PSNR_list=[]
                    for i in [-2,1,4,7,10]:
                        validate_snr=i
                        val_ave_psnr=compute_AvePSNR(model,testloader,validate_snr)
                        PSNR_list.append(val_ave_psnr)
                    ave_PSNR=np.mean(PSNR_list)
                    if ave_PSNR > best_psnr:
                        best_psnr=ave_PSNR
                        print('Find one best model with best PSNR:',best_psnr,' under SNR: ',channel_snr)
                        checkpoint={
                            "model_name":args.model,
                            "net":model.state_dict(),
                            "op":optimizer.state_dict(),
                            "epoch":epoch,
                            "SNR":channel_snr,
                            "Best_PSNR":best_psnr
                        }
                        print(PSNR_list)
                        SNR_path='./checkpoints/SNR_double_T_'+str(channel_snr)  
                        check_dir(SNR_path)      
                        save_path=os.path.join(SNR_path,'Trans_SNR_'+str(channel_snr)+'.pth')
                        torch.save(checkpoint, save_path)
                        print('Saving Model at epoch',epoch,'at',save_path)       

                else:
                    validate_snr=channel_snr
                    val_ave_psnr=compute_AvePSNR(model,testloader,validate_snr)
                    if val_ave_psnr > best_psnr:
                        best_psnr=val_ave_psnr
                        print('Find one best model with best PSNR:',best_psnr,' under SNR: ',validate_snr,'in epoch',epoch)
                        PSNR_list=[]
                        #for i in [1,4,10,16,19]:
                        for i in [-2,1,4,7,10]:
                        #for i in [1]:    
                            ave_PSNR_test=compute_AvePSNR(model,testloader,i)
                            PSNR_list.append(ave_PSNR_test)
                        print('SNR: [-2,1,4,7,10]')
                        print(PSNR_list)
                        checkpoint={
                            "model_name":'SETR',
                            "net":model.state_dict(),
                            "op":optimizer.state_dict(),
                            "epoch":epoch,
                            "SNR":channel_snr,
                            "Best_PSNR":best_psnr
                        }
                        SNR_path='./checkpoints/SNR_T_'+str(channel_snr)  
                        check_dir(SNR_path)      
                        save_path=os.path.join(SNR_path,'SETR_double_iter_'+str(channel_snr)+'.pth')
                        torch.save(checkpoint, save_path)
                        print('Saving Model at epoch',epoch,'at',save_path)       
            