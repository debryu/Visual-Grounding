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


torch.backends.cudnn.benchmark = True 
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

wb = True

#wandb.init(name="FPNv2-Clip-a")
if wb:
    wandb.init(name='RCNN-Clip',project="visual-grounding", entity="unitnais")

#Bounding box data is bottom left x,y top right x,y 
#well apparently no, its top left corner and w,h

path = 'C:/Users/Mateo-drr/Documents/picklesL/sim/'
spath = 'D:/Universidades/Trento/2S/ML/epochs/'
n_epochs = 100
init_lr = 0.00001
clipping_value = 1 #gradient clip
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
criterion = nn.MSELoss()
save_freq =1
batch_size = 16
resize = 128
window_size=32

toTensor = transforms.ToTensor()

stride = 1
kernel_size =8
plot = False

#encoder
numc=4
loaddir = "D:/MachineLearning/RinRUnpix/finalOptions/"+str(numc)+'/UtMq-cprQ2-clipint8-to0-fmod/autoenc00115.pth'

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)


config = {
    "learning_rate": init_lr,
    "batch_size": batch_size,
    "num_epochs": n_epochs,
    "resize":resize
}
wandb.config.update(config)
# Freeze model weights

for param in model.parameters():
    param.requires_grad = False

#model.rpn.requires_grad = True
#model.backbone.requires_grad = True


def loadData(path,split):
    files =  os.listdir(path + split)
    dataset = []
    for file in files:
        file_path = os.path.join(path+split, file)
        
        dataset.append(file_path)
    return dataset
     
    
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
                img = F.interpolate(data['img'], (resize,resize), mode='bicubic', antialias=True)
                blabel = data['Blabel']

        return {'img':img[0],
              'label':label,
              'blabel':blabel[0],
              'bbox':bbox[0]}

#CREATE THE DATALOADER
def create_data_loader_CustomDataset(data, batch_size, eval=False):
    ds = CustomDatasetPP(data=data)

    if not eval:
        return DataLoader(ds, batch_size=batch_size, shuffle=True), len(ds)

    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=False), len(ds)

train_ds = loadData(path, 'train/')
val_ds = loadData(path, 'val/')
test_ds = loadData(path, 'test/')

train_dl, train_length = create_data_loader_CustomDataset(train_ds, batch_size, eval=False)
val_dl, train_length = create_data_loader_CustomDataset(val_ds, batch_size, eval=True)
test_dl, train_length = create_data_loader_CustomDataset(test_ds, batch_size, eval=True)


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

## ############################################################################
def loadCAE(device):
#device = 'cpu'
    autoenc = resConvfmod()
    autoenc = autoenc.to(device)
    #autoenc.load_state_dict(torch.load(loaddir, map_location=torch.device(device)))
    #autoenc.eval()
    return autoenc

class STEFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        lquant = torch.quantize_per_tensor(input, 0.01, 0, dtype=torch.qint8) #3 8 decimals
        unqlat = torch.dequantize(lquant)
        return unqlat

    @staticmethod
    def backward(ctx, grad_output):
        #return F.hardtanh(grad_output)
        return grad_output

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        #self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        #self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        #out = self.RDB2(out)
        #out = self.RDB3(out)
        return out * 0.2 + x

class ResidualDenseBlock_5C_dec(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C_dec, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.ConvTranspose2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.ConvTranspose2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.ConvTranspose2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.ConvTranspose2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.ConvTranspose2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB_dec(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB_dec, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C_dec(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C_dec(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C_dec(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class resConvfmod(nn.Module):
    def __init__(self):
        super(resConvfmod, self).__init__()
        
        self.enc1 = nn.Sequential(conv(3, 16, 3, 1, 1,padding_mode='reflect'),
                                  nn.LeakyReLU(),
                                  nn.PixelUnshuffle(2),
                                  )
        self.enc2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1,padding_mode='reflect'),
                                  nn.PixelUnshuffle(2),
                                  )
        self.RinR = RRDB(nf=128, gc=256)

        #latent
        self.enc3 = nn.Sequential(nn.Conv2d(128, int(numc/4), 3, 1, 1,padding_mode='reflect'),
                                  nn.PixelUnshuffle(2),
                                  )

        self.lat = StraightThroughEstimator()
        
        #DEC
        self.px1 = nn.Sequential(nn.Conv2d(numc, 512, 3, stride=1, padding=1, padding_mode='reflect'), 
                                 nn.PixelShuffle(2)
                                 )
        
        self.RinRdec = RRDB_dec(nf=128, gc=256)

        self.px2 = nn.Sequential(conv(128, 256, 3, stride=1, padding=1,padding_mode='reflect'), 
                                 nn.PixelShuffle(2),
                                 nn.LeakyReLU()
                                 )
        
        
        self.px3 = nn.Sequential(conv(64, 12, 3, stride=1, padding=1,padding_mode='reflect'),
                                 nn.PixelShuffle(2),
                                 )
        
        #CPR
        self.ecpr = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1,padding_mode='reflect'),
                                  nn.PixelUnshuffle(2),
                                  nn.PixelUnshuffle(2),
                                  nn.PixelUnshuffle(2),
                                  nn.Conv2d(1024, numc, 3, 1, 1,padding_mode='reflect'),
                                  )
        self.dcpr = nn.Sequential(nn.Conv2d(numc, 192, 3, stride=1, padding=1, padding_mode='reflect'), 
                                  nn.PixelShuffle(2),
                                  nn.PixelShuffle(2),
                                  nn.PixelShuffle(2),
                                  )
        
        self.ecpr2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1,padding_mode='reflect'),
                                  nn.PixelUnshuffle(2),
                                  nn.PixelUnshuffle(2),
                                  nn.Conv2d(512, numc, 3, 1, 1,padding_mode='reflect')
                                  )
        self.dcpr2 = nn.Sequential(nn.Conv2d(numc, 192, 3, stride=1, padding=1, padding_mode='reflect'), 
                                  nn.PixelShuffle(2),
                                  nn.PixelShuffle(2),
                                  nn.Conv2d(12, 64, 3, stride=1, padding=1, padding_mode='reflect'), 
                                  )
        
    def encoder(self,x):
        oute = self.enc1(x)     
        x = self.ecpr(x) + self.ecpr2(oute)

        oute = self.enc2(oute)
        oute = self.RinR(oute)
        latent = self.enc3(oute)
        cpr = 0.2*x
        unqlat = self.lat(latent + cpr)

        return unqlat
        
    def decoder(self,unqlat):
        outd = self.px1(unqlat)
        x = self.dcpr2(unqlat)
        unqlat = self.dcpr(unqlat)

        outd = self.RinRdec(outd)
        outd = self.px2(outd) + 0.2*x
        out = self.px3(outd) + 0.2*unqlat 
        out = out.clamp(0,1)
        return out
    
    def forward(self,x):
        unqlat = self.encoder(x)
        out = self.decoder(unqlat)
        return out, unqlat
###############################################################################

class BXfinder(nn.Module):
    def __init__(self):
        super(BXfinder, self).__init__()
        
        self.enc1 = nn.Sequential(conv(3, 16, 3, 1, 1,padding_mode='reflect'),
                                  nn.LeakyReLU(inplace=True),
                                  nn.PixelUnshuffle(4),
                                  )
        self.enc2 = nn.Sequential(conv(256,64,3,1,1,padding_mode='reflect'),
                                  nn.LeakyReLU(inplace=True),
                                  nn.PixelUnshuffle(2),
                                  )
        self.enc3 = nn.Sequential(conv(256,32,3,1,1,padding_mode='reflect'),
                                  nn.LeakyReLU(inplace=True),
                                  nn.PixelUnshuffle(2),
                                  conv(128,16,3,1,1,padding_mode='reflect'),                                  
                                  )
        
        self.res = nn.Sequential(nn.LeakyReLU(inplace=True),
                                 conv(16,32,3,1,1,padding_mode='reflect'),
                                 nn.LeakyReLU(inplace=True),
                                 conv(32,16,3,1,1,padding_mode='reflect'),
                                 nn.LeakyReLU(inplace=True),
                                 )
        
        self.lin = nn.Sequential(#nn.Flatten(),
                                 nn.Linear(1024, 2048), #3136
                                 nn.BatchNorm1d(2048),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(2048, 4096),
                                 nn.BatchNorm1d(4096),
                                 nn.LeakyReLU(inplace=True),
                                 )
        
        self.lin2 = nn.Sequential(nn.Linear(4096,4),
                                 nn.LeakyReLU(inplace=True),
                                 )
        
        self.idk = nn.Sequential(conv(3,4,3,1,1,padding_mode='reflect'),
                                 nn.LeakyReLU(inplace=True),
                                 nn.PixelUnshuffle(8),
                                 conv(256,4,3,1,1,padding_mode='reflect'),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Flatten(),
                                 )
        self.down = nn.Sequential(conv(3,4,3,1,1,padding_mode='reflect'),
                                  nn.GELU(),
                                  nn.PixelUnshuffle(2),
                                  conv(16,8,3,1,1,padding_mode='reflect'),
                                  nn.GELU(),
                                  nn.PixelUnshuffle(2),
                                  conv(32,16,3,1,1,padding_mode='reflect'),
                                  nn.GELU(),
                                  nn.PixelUnshuffle(2),
                                  conv(64,32,3,1,1,padding_mode='reflect'),
                                  nn.GELU(),
                                  nn.PixelUnshuffle(2),
                                  conv(128,64,3,1,1,padding_mode='reflect'), #128 32 32
                                  nn.GELU(),
                                  nn.PixelUnshuffle(2),
                                  conv(256,128,3,1,1,padding_mode='reflect'),
                                  nn.GELU(),
                                  nn.PixelUnshuffle(2),
                                  conv(512,256,3,1,1,padding_mode='reflect'),
                                  nn.GELU(),
                                  )
        self.flat = nn.Flatten()
        self.L1 = nn.Sequential(nn.Linear(4096, 4096),
                                nn.GELU(),
                                nn.Linear(4096, 1024),
                                nn.GELU(),
                                nn.Linear(1024, 768),
                                nn.GELU(),
                                )
        
        self.L2 = nn.Sequential(nn.Linear(768, 1024),
                                nn.GELU(),
                                nn.Linear(1024, 768),
                                nn.GELU(),
                                )
        self.L3 = nn.Sequential(nn.Linear(33536, 4096),
                                #nn.Dropout(0.1),
                                nn.GELU(),
                                nn.Linear(4096, 8192),
                                #nn.Dropout(0.3),
                                nn.GELU(),
                                nn.Linear(8192, 8192),
                                #nn.Dropout(0.5),
                                nn.GELU(),
                                nn.Linear(8192, 8192),
                                #nn.Dropout(0.3),
                                nn.GELU(),
                                nn.Linear(8192, 2048),
                                #nn.Dropout(0.1),
                                nn.GELU(),
                                nn.Linear(2048, 4),
                                nn.ReLU()
                                )
        
        self.idk2 = nn.Sequential(conv(320,128,3,1,1,padding_mode='reflect'),
                                  nn.GELU(),
                                  conv(128,64,3,1,1,padding_mode='reflect'),
                                  nn.GELU(),
                                  conv(64,32,3,1,1,padding_mode='reflect'),
                                  nn.GELU())
        #self.mha = nn.Sequential(nn.Embedding(1536,1024),
        #                         nn.MultiheadAttention(embed_dim=1024, num_heads=8))
        self.up5 = nn.Sequential(conv(256, 256,3,1,1,padding_mode='reflect'),
                                nn.GELU(),
                                nn.PixelShuffle(2))
        self.up4 = nn.Sequential(conv(256+64, 256,3,1,1,padding_mode='reflect'),
                                nn.GELU(),
                                nn.PixelShuffle(2))
        
        
        self.idk = model.backbone
        
    def forward(self,x, bertx):
        #x = self.down(x)
        x = self.idk(x)
        
        
        x1 = x.popitem(last=False)[1]
        x2 = x.popitem(last=False)[1]
        x3 = x.popitem(last=False)[1]
        x4 = x.popitem(last=False)[1]
        x5 = x.popitem(last=False)[1]
        
        x1.requires_grad = True
        x2.requires_grad = True
        x3.requires_grad = True
        x4.requires_grad = True
        x5.requires_grad = True
        
        x5 = self.up5(x5) #256,2,2 -> 64,4,4
        x4 = torch.cat((x4,x5), dim=1) # 256+64,4,4
        x4 = self.up4(x4) #320,4,4 -> 64,8,8
        x3 = torch.cat((x3,x4), dim=1) # 256+64,8,8
        x3 = self.up4(x3) #320,8,8 -> 64,16,16
        x2 = torch.cat((x2,x3), dim=1) # 256+64,16,16
        x2 = self.up4(x2) #320,16,16 -> 64,32,32
        x1 = torch.cat((x1,x2), dim=1) # 256+64,32,32

        
        x = self.idk2(x1)
        x = self.flat(x)
        #print(x.size())
        #x = self.flat(x)
        #print(x.size())
        #x = self.L1(x)
        #print(x.size())
        #x = self.L2(x)
        #print(x.size(), bertx.size())
        #bertx = self.L2(bertx)
        x = torch.cat((x, bertx), dim=1)
        #x = self.mha(x)
        #print(x.size(), bertx.size())
        x = self.L3(x)
        #print(x.size())
        return x#*(resize-1)#torch.clip(x,0,1)*resize
  
#init the autoencoder    
#compress = loadCAE(device)

bxfinder = BXfinder();
bxfinder.to(device)
model.backbone.to(device)
optimizer = torch.optim.AdamW(bxfinder.parameters(), lr=init_lr)  

#model.rpn = nn.Sequential()
#model.roi_heads = nn.Sequential()

def final_loss(outputs, bbox):
    loss = 0
    for i in range(0,4):
        loss += criterion(outputs.transpose(1,0)[i], bbox.transpose(1,0)[i])
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
        l1 = tvo.generalized_box_iou_loss(outputs,bbox,reduction='mean')
    
    l2 = loss/4
    
    return torch.sqrt(l1*l2)


for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    val_loss= 0.0
    print(epoch)
    
    bxfinder.train()
    #compress.train()
    i=0
    for bi, data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
    
        images,label,bbox = data['img'], data['label'][0], data['bbox']
        
        images = images.to(device)
        bbox = bbox.to(device)
        bbox[:,2] = bbox[:,0]+bbox[:,2]
        bbox[:,3] = bbox[:,1]+bbox[:,3]
        
        #mask = torch.mean(images, dim=1, keepdim=True)
        
        
        #recImgs, bertLabels = preprocessing(images, label)
        recImgs = images
        bertLabels = data['blabel']
        
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
        
        
        if wb:
            wandb.log({'tloss': loss})
        #if i%10 == 0:
        #    print(outputs.transpose(0,1)[0][0:4], bbox.transpose(0,1)[0][0:4])
        
        i+=1
    
    train_loss = train_loss/len(train_dl)
    print('E: {} T Loss: {:.3f}'.format(epoch, train_loss) + " %" + "{:.3}".format(np.exp(-abs(train_loss))*100))
    print(outputs[0], bbox[0])
    if epoch%save_freq == 0:
        try:
            torch.save(bxfinder.state_dict(), spath + 'epoch{0:05d}.pth'.format(epoch))
        except Exception as e:
            print("An error occurred:", e)
            
        if wb:
            wandb.save(path + 'wandb/wandb{0:05d}.pth'.format(epoch))


    bxfinder.eval()
    #compress.eval()
    with torch.no_grad():
        for bi, data in tqdm(enumerate(val_dl), total=int(len(val_ds)/val_dl.batch_size)):
        #for data in val_dl:
            images,label,bbox = data['img'], data['label'], data['bbox']
            images = images.to(device)
            bbox = bbox.to(device)
            bbox[:,2] = bbox[:,0]+bbox[:,2]
            bbox[:,3] = bbox[:,1]+bbox[:,3]
            
            recImgs = images
            bertLabels = data['blabel']
           
           
            #latent = compress.encoder(recImgs) 
           
            outputs = bxfinder(recImgs, bertLabels.to(device))
            
            loss = loss = final_loss(outputs, bbox)
            val_loss += loss.item()*images.size(0)
            
            
        val_loss = val_loss/len(val_dl)
        print('E: {} V Loss: {:.3f}'.format(epoch, val_loss) + " %" + "{:.3}".format(np.exp(-abs(val_loss))*100))
        print(outputs[0], bbox[0])
        
        if wb:
            wandb.log({'Train loss': train_loss, 'Validation Loss': val_loss})
        
#'''