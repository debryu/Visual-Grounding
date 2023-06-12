 # -- coding: utf-8 --
"""
Created on Thu Apr 20 10:24:49 2023

@author: Mateo-drr
"""


import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as tt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from torchvision import transforms
import torchvision.ops as tvo
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import wandb
from PIL import Image
from transformers import BertTokenizer, BertModel
import time
import datetime

wb = True
#wandb.init(name="FPNv2-Clip-a")


#Bounding box data is bottom left x,y top right x,y 
#well apparently no, its top left corner and w,h
#path = 'D:/MachineLearning/datasets/refcocog/refcocog/all/FINAL/'
path = 'C:/Users/Mateo-drr/Documents/FINAL/'
spath = 'D:/Universidades/Trento/2S/ML/epochs/'
n_epochs = 100
init_lr = 5e-5#0.00005
clipping_value = 1 #gradient clip
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
criterion = nn.MSELoss()
save_freq =5
batch_size = 32
resize = 256
window_size=32
plot = False

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, trainable_backbone_layers=5)
model2 = fasterrcnn_resnet50_fpn_v2(weights=weights, trainable_backbone_layers=5)

# Freeze model weights

#for param in model.parameters():
#    param.requires_grad = False

#model.rpn.requires_grad = True
#model.backbone.requires_grad = True


def loadData(path,split):
    files =  os.listdir(path + split)
    dataset = []
    for file in files:
        file_path = os.path.join(path+split, file)
        
        dataset.append(file_path)
    return dataset
     
class CustomDataset(Dataset):

    def __init__(self, data):
    #WHAT DO WE PUT HERE?
        self.data = data

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):
    #TAKE ONE ITEM FROM THE DATASET
    
        
        with open(self.data[idx], 'rb') as file:             
            data = pickle.load(file)
            if True: 
                label = [data['label']['sentences'][0]['raw']] #take the first label raw
                annotation = data['annotation']['bbox']
                img = Image.frombytes(data['img']['mode'],
                                      data['img']['size'],
                                      data['img']['pixels'])
                
                img = np.transpose(np.array(img, dtype=np.float32)/255)
                if img.shape[0] != 3:
                    img = np.repeat(img[np.newaxis, :,:],3,axis=0)
                img = torch.tensor(img)
                transform = tt.Resize((resize,resize), interpolation=tt.InterpolationMode.BICUBIC, antialias=True)
                img = transform(img)
                #img = F.Resize(img, 224,interpolation=tt.InterpolationMode.BICUBIC)
                rsize = [data['img']['size'][0]/img.size()[1],
                         data['img']['size'][1]/img.size()[2]]
                bbox = [annotation[0]/rsize[0], annotation[1]/rsize[1],
                        annotation[2]/rsize[0], annotation[3]/rsize[1]]

        #decode = self.tokenizer.convert_ids_to_tokens(encoding_text['input_ids'].flatten())

        return {'img':img,
              'label':label,
              'bbox':torch.tensor(bbox)}
    
class CustomDatasetPP(Dataset):

    def __init__(self, data):
    #WHAT DO WE PUT HERE?
        self.data = data

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):
    #TAKE ONE ITEM FROM THE DATASET
    
        
        with open(self.data[idx], 'rb') as file:             
            data = pickle.load(file)
            if True:#data['img']['mode'] == 'RGB': TODO   
                label = data['label'] #take the first label raw
                bbox = data['bbox']*resize/512
                #img = F.interpolate(data['mix'], (resize,resize), mode='bicubic', antialias=True)
                img = F.interpolate(data['img'], (resize,resize), mode='bicubic', antialias=True)
                blabel = data['Blabel']
                
                
                #cj = tt.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 5))
                #img = cj(img)
                mean = [0.5, 0.5, 0.5]  # Mean values for each channel
                std = [0.5, 0.5, 0.5]   # Standard deviation values for each channel

                normalize = transforms.Normalize(mean, std)
                img = normalize(img)
                
        img.requires_grad=False
        blabel.requires_grad=False
        bbox.requires_grad=False

        return {'img': img[0],
              'label': label,
              'blabel':blabel[0],
              'bbox':bbox[0]}

def main():
#if True:
    
    torch.backends.cudnn.benchmark = True 
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    if wb:
        wandb.init(name='mod4v2-512b8',project="visual-grounding", entity="unitnais")

    #CREATE THE DATALOADER
    def create_data_loader_CustomDataset(data, batch_size, eval=False):
        ds = CustomDatasetPP(data=data)
    
        if not eval:
            return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True), len(ds)
    
        else:
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2), len(ds)
    
    train_ds = loadData(path, 'train/')
    val_ds = loadData(path, 'val/')
    test_ds = loadData(path, 'test/')
    
    print(len(train_ds), len(val_ds), len(test_ds))
    
    train_dl, train_length = create_data_loader_CustomDataset(train_ds, batch_size, eval=False)
    val_dl, train_length = create_data_loader_CustomDataset(val_ds, batch_size, eval=True)
    test_dl, train_length = create_data_loader_CustomDataset(test_ds, 1, eval=True)
    
    
    #'''
    for batch in train_dl:
        for i,image in enumerate(batch['img']):
            print(image.shape)
            image = image.squeeze(0).permute(2,1,0)
            fig, ax = plt.subplots()
    
            ax.imshow(image)
            plt.title(batch['label'][0][i])
            x_down = batch['bbox'][i][0].item()
            y_down = batch['bbox'][i][1].item()
            w = batch['bbox'][i][2].item()
            h = batch['bbox'][i][3].item()
            print(x_down,y_down,w,h)
            rect = plt.Rectangle((x_down, y_down), w, h, linewidth=1, edgecolor='r', facecolor='none',alpha=0.3,color='r')
            ax.add_patch(rect)
            plt.show()
            break
            
        #print(batch['img'].shape)
        #plt.imshow(item['img'])
        #print(batch['label'])
        print(batch['bbox'])
        break
    
    #'''
    
    def conv(ni, nf, ks=3, stride=1, padding=1, **kwargs):
        _conv = nn.Conv2d(ni, nf, kernel_size=ks,stride=stride,padding=padding, **kwargs)
        nn.init.kaiming_normal_(_conv.weight, mode='fan_out')
        return _conv
    
    class BXfinder(nn.Module):
        def __init__(self):
            super(BXfinder, self).__init__()
            
            self.flat = nn.Flatten()
            self.L1 = nn.Sequential(conv(17,8,3,1,1,padding_mode='reflect'),
                                    nn.Mish(inplace=True),
                                    conv(8,3,3,1,1,padding_mode='reflect'),
                                    nn.Mish(inplace=True),
                                    )
            
            self.L2 = nn.Sequential(nn.Linear(768, 2048),
                                    nn.Mish(inplace=True),
                                    nn.Linear(2048, int(resize/4)*int(resize/4)*3),
                                    nn.Mish(inplace=True),
                                    )
            self.L3 = nn.Sequential(nn.Linear(int(resize/8)*int(resize/8)*9, 4096),
                                    #nn.Dropout(0.1),
                                    nn.Mish(inplace=True),
                                    nn.Linear(4096, 8192),
                                    #nn.Dropout(0.1),
                                    nn.Mish(inplace=True),
                                    nn.Linear(8192, 8192),
                                    #nn.Dropout(0.1),
                                    #nn.Mish(inplace=True),
                                    #nn.Linear(8192, 8192),
                                    #nn.Dropout(0.1),
                                    #nn.Mish(inplace=True),
                                    #nn.Linear(8192, 8192),
                                    #nn.Dropout(0.1),
                                    nn.Mish(inplace=True),
                                    nn.Linear(8192, 2048),
                                    #nn.Dropout(0.1),
                                    nn.Mish(inplace=True),
                                    nn.Linear(2048, 512),
                                    nn.Mish(inplace=True),
                                    nn.Linear(512, 4),
                                    nn.Sigmoid()
                                    )
            
            self.idk2 = nn.Sequential(conv(320,64,3,1,1,padding_mode='reflect'),
                                      nn.LeakyReLU(),
                                      conv(64,32,3,1,1,padding_mode='reflect'),
                                      nn.LeakyReLU(),
                                      conv(32,9,3,1,1,padding_mode='reflect'),
                                      nn.LeakyReLU())
            
            self.r1 = nn.Conv2d(320,128,3,1,1,padding_mode='reflect')
                                    
            self.r2 = nn.Sequential(nn.Mish(inplace=True),
                                    conv(128,128,3,1,1,padding_mode='reflect'),
                                    nn.Mish(inplace=True),
                                    )
            self.r3 = nn.Conv2d(128,64,3,1,1,padding_mode='reflect')
            self.r4 = nn.Sequential(nn.Mish(inplace=True),
                                    conv(64,64,3,1,1,padding_mode='reflect'),
                                    nn.Mish(inplace=True),)
            self.r5 = nn.Conv2d(64,32,3,1,1,padding_mode='reflect')
            self.r6 = nn.Sequential(nn.Mish(inplace=True),
                                    conv(32,32,3,1,1,padding_mode='reflect'),
                                    nn.Mish(inplace=True),)
            self.r7 = nn.Conv2d(32,9,3,1,1,padding_mode='reflect')
            
            
            #self.mha = nn.Sequential(nn.Embedding(1536,1024),
            #                         nn.MultiheadAttention(embed_dim=1024, num_heads=8))
            self.up5 = nn.Sequential(conv(256, 256,3,1,1,padding_mode='reflect'),
                                    nn.Mish(inplace=True),
                                    nn.PixelShuffle(2))
            self.up4 = nn.Sequential(conv(256+64, 256,3,1,1,padding_mode='reflect'),
                                    nn.Mish(inplace=True),
                                    nn.PixelShuffle(2))
            
            
            self.idk = model.backbone
            self.lol = model2.backbone
            
        def forward(self,x, bertx):
            x = self.idk(x)
            
            
            x1 = x.popitem(last=False)[1]
            x2 = x.popitem(last=False)[1]
            x3 = x.popitem(last=False)[1]
            x4 = x.popitem(last=False)[1]
            x5 = x.popitem(last=False)[1]
            #print(x5.shape, x4.shape, x3.shape, x2.shape, x1.shape)
            
            x1.requires_grad_(True)
            x2.requires_grad_(True)
            x3.requires_grad_(True)
            x4.requires_grad_(True)
            x5.requires_grad_(True)
            
            x5 = self.up5(x5) #256,2,2 -> 64,4,4
            x4 = torch.cat((x4,x5), dim=1) # 256+64,4,4
            x4 = self.up4(x4) #320,4,4 -> 64,8,8
            x3 = torch.cat((x3,x4), dim=1) # 256+64,8,8
            x3 = self.up4(x3) #320,8,8 -> 64,16,16
            x2 = torch.cat((x2,x3), dim=1) # 256+64,16,16
            x2 = self.up4(x2) #320,16,16 -> 64,32,32
            x1 = torch.cat((x1,x2), dim=1) # 256+64,32,32
            #print( x3.shape)
    
            #RES BLOCKS ENCODER
            #x = self.idk2(x1) #output -> [9,32,32]
            x = self.r1(x1)
            x = self.r2(x) + x
            x = self.r3(x)
            x = self.r4(x) + x
            x = self.r5(x)
            x = self.r6(x) + x
            x = self.r7(x)
            #print(x.shape)
            
            #idk
            top_left = x[:, :3, :, :]
            top_right = x[:, 3:6, :, :]
            bottom_left = x[:, 6:, :, :]
            '''
            x = self.r1(x1)
            x = self.r2(x) + x
            x = self.r3(x)
            x = self.r4(x) + x
            x = self.r5(x)
            x = self.r6(x) + x
            '''
            #print(bertx.shape)
            bertx = self.L2(bertx) #768 -> 1024
            #print(bertx.shape)
            bertx = bertx.reshape(-1, 3, int(resize/4), int(resize/4)) 
            #print(bertx.shape)
            xx = torch.zeros((bertx.shape[0], 3, int(resize/2), int(resize/2)), device=torch.device(device))#.to(device)
            xx[:, :, :int(resize/4), :int(resize/4)] = top_left
            xx[:, :, :int(resize/4), int(resize/4):] = top_right
            xx[:, :, int(resize/4):, :int(resize/4)] = bottom_left
            xx[:, :, int(resize/4):, int(resize/4):] = bertx
            
            #x = torch.cat((x, bertx), dim=2) # [batch,16,64,32]
            #print(xx.shape, xx.requires_grad)
            #x = self.L1(xx) # [17,64,32] -> [3,64,32]
            #print(x.shape)
            x = self.lol(xx)
            
            x1 = x.popitem(last=False)[1] #[32,256,8,8]
            x2 = x.popitem(last=False)[1]
            x3b = x.popitem(last=False)[1]
            x4 = x.popitem(last=False)[1]
            #x5 = x.popitem(last=False)[1]
            #print(x5.shape, x4.shape, x3.shape, x2.shape, x1.shape)
            
            x4 = self.up5(x4) #256,2,2 -> 64,4,4
            x3b = torch.cat((x3b,x4), dim=1) # 256+64,4,4
            x3b = self.up4(x3b) #320,4,4 -> 64,8,8
            x2 = torch.cat((x2,x3b), dim=1) # 256+64,8,8
            x2 = self.up4(x2) + x3#320,8,8 -> 64,16,16
            #print(x2.shape, x3.shape)
            x1 = torch.cat((x1,x2), dim=1) # 256+64,16,16
            #x2 = self.up4(x2) #320,16,16 -> 64,32,32
            #x1 = torch.cat((x1,x2), dim=1) # 256+64,32,32
            
            # x = self.r1(x1)
            # x = self.r2(x) + x
            # x = self.r3(x)
            # x = self.r4(x) + x
            # x = self.r5(x)
            # x = self.r6(x) + x
            # x = self.r7(x)  

            x = self.idk2(x1) # 320,16,16 -> 9,16,16
            
            #print(x1.shape)
            x = self.flat(x)
            #print(x1.shape)
            x = self.L3(x) #

            return x*(resize-1)
      
    #init the autoencoder    
    #compress = loadCAE(device)
    
    bxfinder = BXfinder();
    if wb:
        config = {
            "learning_rate": init_lr,
            "batch_size": batch_size,
            "num_epochs": n_epochs,
            "resize":resize,
            "model":bxfinder
        }
        wandb.config.update(config)
    
    bxfinder.to(device)
    model.backbone.to(device)
    model2.backbone.to(device)
    optimizer = torch.optim.AdamW(bxfinder.parameters(), lr=init_lr)  
    
    def final_loss(outputs, bbox):
        loss = 0
        for i in range(0,4):
            loss += criterion(outputs.transpose(1,0)[i], bbox.transpose(1,0)[i])
            #print(criterion(outputs.transpose(1,0)[i], bbox.transpose(1,0)[i]))

        #return loss/4
        '''
        boxA,boxB = outputs.clone(), bbox.clone()
        boxA[:,2] = boxA[:,0]+boxA[:,2]
        boxA[:,3] = boxA[:,1]+boxA[:,3]
        boxB[:,2] = boxB[:,0]+boxB[:,2]
        boxB[:,3] = boxB[:,1]+boxB[:,3]
        '''
        #print(tvo.generalized_box_iou_loss(outputs,bbox))
        neg = tvo.generalized_box_iou_loss(outputs,bbox)
        if (neg < 0).any():
            mask = neg >= 0
            pos = torch.masked_select(neg, mask)
            l1 = torch.mean(pos)
        else:
            l1 = tvo.generalized_box_iou_loss(outputs,bbox,reduction='sum')
        
        l2 = loss
        
        #max error = (l2*l1*100)/(bbox.size(0)*((resize*np.sqrt(2))**2)*4 *2*bbox.size(0))
        #print((l2*l1))
        return (l1*l2)/(resize*bbox.size(0)) #torch.sqrt(l1*l2)
    
    elapsed_time=0
    for epoch in range(1, n_epochs+1):
        start_time = time.time()
        train_loss = 0.0
        val_loss= 0.0
        print(epoch, 'ETA: ' + str(datetime.timedelta(seconds = elapsed_time*n_epochs+1-epoch)))
        
        bxfinder.train()
        #compress.train()
        i=0
        iou = []
        for bi, data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
        
            images,label,bbox = data['img'], data['label'][0], data['bbox']
            #label = label.to(device)
            images = images.to(device)
            bbox = bbox.to(device)
            bbox[:,2] = bbox[:,0]+bbox[:,2]
            bbox[:,3] = bbox[:,1]+bbox[:,3]
            
            #mask = torch.mean(images, dim=1, keepdim=True)
            
            
            #recImgs, bertLabels = preprocessing(images, label)
            
            
            #bertLabels = torch.zeros(len(label),768)
            '''
            with torch.no_grad():
                bertLabels = torch.zeros(len(label),768)
                for i in range(0,len(label)):
                    encoded_input = tokenizer(label[i], return_tensors='pt').to(device)
                    output = bertM(**encoded_input).last_hidden_state[0][0] #768 size
                    bertLabels[i] = output
            '''
            
            recImgs = images
            bertLabels = data['blabel']
            #bertLabels.requires_grad = True
            images.requires_grad=True
            bertLabels.requires_grad=True
            bbox.requires_grad=True
            
            #recImgs,bertLabels,label,bbox =recImgs.to('cpu'),bertLabels.to('cpu'),label,bbox.to('cpu') 
            #with open(path+'sim/test/'+test_ds[i][-28:], 'wb') as f:
            #    pickle.dump({'img': recImgs, 'Blabel':bertLabels, 'label':label, 'bbox':bbox}, f)
                
            #i+=1    
            
            #bb = model.backbone(recImgs)
            #aa = model.fpn(bb)
            
            #latent = compress.encoder(recImgs)
    #'''    
            #latent.requires_grad = True
         
            outputs = bxfinder(recImgs, bertLabels.to(device))
            loss = final_loss(outputs, bbox)
            #print(outputs)
            optimizer.zero_grad()
            loss.backward()        
            torch.nn.utils.clip_grad_norm_(bxfinder.parameters(), clipping_value)
            optimizer.step()
            train_loss += loss.item()*images.size(0)
            
            
            for j in range(0,images.size(0)):    
                temp = tvo.box_iou(bbox[j].unsqueeze(0),outputs[j].unsqueeze(0)).to('cpu')
                iou.append(temp.item())
            
            if wb:
                wandb.log({'tloss': loss})
            #if i%10 == 0:
            #    print(outputs.transpose(0,1)[0][0:4], bbox.transpose(0,1)[0][0:4])
            
            i+=1
            #break
        
        train_loss = train_loss/len(train_dl)
        print('E: {} T Loss: {:.3f}'.format(epoch, train_loss) + " %" + "{:.3}".format(np.exp(-abs(train_loss))*100))
        print(np.array(iou).mean())
        if epoch%save_freq == 0:
            try:
                torch.save(bxfinder.state_dict(), spath + 'epoch{0:05d}.pth'.format(epoch))
            except Exception as e:
                print("An error occurred:", e)
                
            if wb:
                wandb.save(path + 'wandb/wandb{0:05d}.pth'.format(epoch))
    
        if wb:
            #wandb.log({'Train iou': iou})
            wandb.log({'t iou line': np.array(iou).mean()})
    
        bxfinder.eval()
        #compress.eval()
        iou=[]
        with torch.no_grad():
            for bi, data in tqdm(enumerate(val_dl), total=int(len(val_ds)/val_dl.batch_size)):
            #for data in val_dl:
                images,label,bbox = data['img'], data['label'][0], data['bbox']
                
                #label = label.to(device)
                images = images.to(device)
                bbox = bbox.to(device)
                bbox[:,2] = bbox[:,0]+bbox[:,2]
                bbox[:,3] = bbox[:,1]+bbox[:,3]
                
                recImgs = images
                
                #bertLabels = torch.zeros(len(label),768)
                '''
                for i in range(0,len(label)):
                    encoded_input = tokenizer(label[i], return_tensors='pt').to(device)
                    output = bertM(**encoded_input).last_hidden_state[0][0] #768 size
                    bertLabels[i] = output
                #bertLabels = data['blabel']
                '''
               
                #latent = compress.encoder(recImgs) 
                recImgs = images
                bertLabels = data['blabel']
                
                outputs = bxfinder(recImgs, bertLabels.to(device))
                
                loss = loss = final_loss(outputs, bbox)
                val_loss += loss.item()*images.size(0)
                
                for j in range(0,images.size(0)):    
                    temp = tvo.box_iou(bbox[j].unsqueeze(0),outputs[j].unsqueeze(0)).to('cpu')
                    iou.append(temp.item())
                
                
            val_loss = val_loss/len(val_dl)
            print('E: {} V Loss: {:.3f}'.format(epoch, val_loss) + " %" + "{:.3}".format(np.exp(-abs(val_loss))*100))
            print(np.array(iou).mean())
            print(outputs[0],'\n', bbox[0])
            print(outputs[1],'\n', bbox[1])
            print(outputs[2],'\n', bbox[2])
            
            
            if wb:
                wandb.log({'Train loss': train_loss, 'Validation Loss': val_loss})
                #wandb.log({'Validation iou': iou})
                wandb.log({'v iou line': np.array(iou).mean()})
        
        elapsed_time = time.time() - start_time
    #'''
    #load best config
    #ldr = spath + 'epoch00100.pth'
    #bxfinder.load_state_dict(torch.load(ldr, map_location=torch.device(device)))
    
    with torch.no_grad():
        
        #Calculate test IOU\
        iou = []
        for data in test_dl:
            images,label,bbox = data['img'], data['label'][0], data['bbox']
            bertLabels = data['blabel']
            images = images.to(device)
            bbox = bbox.to(device)
            bbox[:,2] = bbox[:,0]+bbox[:,2]
            bbox[:,3] = bbox[:,1]+bbox[:,3]
            
            outputs = bxfinder(images, bertLabels.to(device))
            temp = tvo.box_iou(bbox,outputs).to('cpu')
            iou.append(temp.item())
        print(np.array(iou).mean())
    
if __name__ == "__main__":
    main()