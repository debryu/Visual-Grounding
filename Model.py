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
from PIL import Image
from tqdm import tqdm
import pickle
import copy
from matplotlib.patches import Rectangle
import clip
import matplotlib.patches as patches
from transformers import BertTokenizer, BertModel

torch.backends.cudnn.benchmark = True 
torch.set_num_threads(8)


#Bounding box data is bottom left x,y top right x,y 

path = 'C:/Users/Mateo-drr/Documents/picklesL/'
n_epochs = 2
init_lr = 0.0009
clipping_value = 1 #gradient clip
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
criterion = nn.MSELoss()
save_freq =1
batch_size = 128
resize = 512
window_size=32


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
            if True:#data['img']['mode'] == 'RGB': TODO   
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

#CREATE THE DATALOADER
def create_data_loader_CustomDataset(data, batch_size, eval=False):
    ds = CustomDataset(data=data)

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


#yoloM = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo='check') #ans yes
#yoloM.to(device)
    
clipM, preprocess = clip.load("RN50", device=device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertM = BertModel.from_pretrained("bert-base-uncased")


for batch in train_dl:
    for i,image in enumerate(batch['img']):
        print(image.shape)
        image = image.permute(2,1,0)
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
        
    print(batch['img'].shape)
    #plt.imshow(item['img'])
    print(batch['label'])
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
        self.L1 = nn.Sequential(nn.Linear(16384, 4096),
                                nn.GELU(),
                                nn.Linear(4096, 1024),
                                nn.GELU(),
                                nn.Linear(1024, 768),
                                nn.GELU(),
                                )
        
        self.L2 = nn.Sequential(nn.Linear(768, 256),
                                nn.GELU(),
                                nn.Linear(256, 128),
                                nn.GELU(),
                                )
        self.L3 = nn.Sequential(nn.Linear(128, 32),
                                nn.GELU(),
                                nn.Linear(32, 4))
        
        self.idk2 = nn.Linear(3136,1024)
        
    def forward(self,x, bertx):
        x = self.down(x)
        x = self.flat(x)
        x = self.L1(x)
        
        x = self.L2(x)
        bertx = self.L2(bertx)
        
        x = self.L3(x + bertx)
        return x
  
bxfinder = BXfinder();
bxfinder.to(device)
optimizer = torch.optim.AdamW(bxfinder.parameters(), lr=init_lr)  

def final_loss(outputs, bbox):
    loss = 0
    for i in range(0,4):
        loss += criterion(outputs.transpose(1,0)[i], bbox.transpose(1,0)[i])
    return loss/4

def windowize(image,window_size):
        windows  = image.unsqueeze(0).permute(0,2,3,1)
        #print(windows.shape)
        windows = windows.unfold(1,window_size,window_size)
        #print(windows.shape)
        x = windows.shape[1]
        windows = windows.unfold(2,window_size,window_size)
        #print(windows.shape)
        y = windows.shape[2]
        #print(y)
        '''
        fig, ax = plt.subplots(x, y, figsize=(2*y, 2*x))
        for i in range(y):
                for j in range(x):
                        ax[j,i].imshow(windows[0,i,j].permute(2,1,0))
                        ax[j,i].axis('off')

        fig.tight_layout()
        plt.show()
        '''
        return windows,x,y

def computeHeatmap(windows,window_size,model,text,device):
    stride = 1
    kernel_size = 6
    #text = 'a pair of white shoes'
    #txt_embedding = text2embedding(text)
    text_input = clip.tokenize([text]).to(device)

    heatmap = torch.zeros(windows.shape[1], windows.shape[2])
    #Count how many time each window is used as a kernel, in order to normalize the results
    # in the heatmap score
    number_of_uses = torch.ones(windows.shape[1], windows.shape[2])

    for i in range(0, windows.shape[1] - kernel_size + 1, stride):
            for j in range(0, windows.shape[2] - kernel_size + 1, stride):
                    #Initialize the canvas as a random noise image 
                    canvas = torch.zeros(window_size*kernel_size, window_size*kernel_size, 3).to(device)
                    #print(canvas.shape)

                    kernel = windows[0,i:i+kernel_size,j:j+kernel_size]
                    #print(kernel.shape)
                    #Add each window of the kernel to the canvas
                    for k in range(kernel.shape[1]):
                            for l in range(kernel.shape[0]):
                                    canvas[k*window_size:(k+1)*window_size, l*window_size:(l+1)*window_size] += kernel[k,l].permute(1,2,0)

                    #plt.imshow(canvas)
                    #plt.show()
                    #print('original')
                    #print(canvas.shape)
                    canvas = canvas.permute(2,0,1)
                    #print(canvas.shape)
                    canvas_PIL = tt.ToPILImage()(canvas)
                    
                    canvas_input = preprocess(canvas_PIL).unsqueeze(0).to(device)
                    
                    #Clip classification
                    logits_per_image, logits_per_text = model(canvas_input, text_input)
                    similarity = logits_per_image.to('cpu').item()
                            

                    heatmap[i:i+kernel_size,j:j+kernel_size] += similarity
                    number_of_uses[i:i+kernel_size,j:j+kernel_size] += 1
                    #print("kernel similarity:", similarity)  # prints: [[0.9927937  0.00421068 0.00299572]]

    heatmap = heatmap / number_of_uses
    return heatmap

def getHeatmap(windows,hm_scores,x,y, plot = False):
        augmented_scores = (hm_scores-hm_scores.mean()).to(device)
        #if lvl > 0:   
        #    augmented_scores = augmented_scores-augmented_scores.mean()
            #augmented_scores = np.clip(augmented_scores-augmented_scores.mean(), 0, np.inf)
        augmented_scores = (augmented_scores - augmented_scores.min())/(augmented_scores.max() - augmented_scores.min())

        #Make a copy of the tensor
        hm = windows.squeeze(0)
        #print(hm.shape)
        hm = hm.permute(3,4,0,1,2)
        hm = hm.permute(0,1,4,2,3)
        hm = hm*augmented_scores
        hm = hm.permute(3,4,2,0,1).unsqueeze(0)
        if(plot):
                fig, ax = plt.subplots(x, y, figsize=(0.5*y, 0.5*x))
                for i in range(y):
                        for j in range(x):
                                ax[j,i].imshow(hm[0,i,j].permute(2,1,0))
                                ax[j,i].axis('off')
                                ax[j,i].set_aspect('equal')

                fig.subplots_adjust(wspace=0.0,hspace=0.0)
                plt.show()
        return hm
    

  
for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    val_loss= 0.0
    print(epoch)
    
    bxfinder.train()
    for bi, data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
        images,label,bbox = data['img'], data['label'][0], data['bbox']
        images = images.to(device)
        bbox = bbox.to(device)
        
        #Chunking the images:
        heatmaps = torch.zeros(batch_size,16,16,3,window_size,window_size)

        bertLabels = torch.zeros(batch_size,768)
        for i,image in enumerate(images):
            windowed_images,x_size,y_size = windowize(image,window_size)
            hm_scores = computeHeatmap(windowed_images,window_size,clipM,label[i],device)
            #getHeatmap(windowed_images,hm_scores,x_size,y_size)
            heatmaps[i] = getHeatmap(windowed_images,hm_scores,x_size,y_size,plot = False)
            
        
            #Encode label with bert
            encoded_input = tokenizer(label[i], return_tensors='pt')
            output = bertM(**encoded_input).last_hidden_state[0][0] #768 size
            bertLabels[i] = output
        #outputs = torch.tensor(outputs)
        
        
        #Rebuild chunks into images
        recImgs = heatmaps.permute(0,3,2,5,1,4) # [1,16a,16b,3,32,32] -> [1,3,16b,32,16a,32]
        recImgs = recImgs.reshape(batch_size,3,512,512)
        #print(output.size(), heatmaps.size(), bertLabels.size(), recImgs.size())

       
        outputs = bxfinder(images, bertLabels.to(device))
        loss = final_loss(outputs, bbox)
        #print(outputs)
        optimizer.zero_grad()
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(bxfinder.parameters(), clipping_value)
        optimizer.step()
        train_loss += loss.item()*images.size(0)
        #break
      
    train_loss = train_loss/len(train_dl)
    print('E: {} T Loss: {:.3f}'.format(epoch, train_loss) + " %" + "{:.3}".format(np.exp(-abs(train_loss))*100))
    print(outputs.transpose(0,1)[0][0:3], bbox.transpose(0,1)[0][0:3])
    if epoch%save_freq == 0:
        torch.save(bxfinder.state_dict(), path + 'epochs/epoch{0:05d}.pth'.format(epoch))


    bxfinder.eval()
    with torch.no_grad():
        for bi, data in tqdm(enumerate(val_dl), total=int(len(val_ds)/val_dl.batch_size)):
        #for data in val_dl:
            images,label,bbox = data['img'], data['label'], data['bbox']
            images = images.to(device)
            bbox = bbox.to(device)
            
            #Chunking the images:
            heatmaps = torch.zeros(batch_size,16,16,3,window_size,window_size)

            bertLabels = torch.zeros(batch_size,768)
            for i,image in enumerate(images):
                windowed_images,x_size,y_size = windowize(image,window_size)
                hm_scores = computeHeatmap(windowed_images,window_size,clipM,label[i],device)
                #getHeatmap(windowed_images,hm_scores,x_size,y_size)
                heatmaps[i] = getHeatmap(windowed_images,hm_scores,x_size,y_size,plot = False)
                
            
                #Encode label with bert
                encoded_input = tokenizer(label[i], return_tensors='pt')
                output = bertM(**encoded_input).last_hidden_state[0][0] #768 size
                bertLabels[i] = output
            #outputs = torch.tensor(outputs)
            
            
            #Rebuild chunks into images
            recImgs = heatmaps.permute(0,3,2,5,1,4) # [1,16a,16b,3,32,32] -> [1,3,16b,32,16a,32]
            recImgs = recImgs.reshape(batch_size,3,512,512)
            print(output.size(), heatmaps.size(), bertLabels.size(), recImgs.size())
            
            outputs = bxfinder(images, bertLabels.to(device))
            loss = final_loss(outputs, bbox)
            val_loss += loss.item()*images.size(0)
            #break
        val_loss = val_loss/len(val_dl)
        print('E: {} V Loss: {:.3f}'.format(epoch, val_loss) + " %" + "{:.3}".format(np.exp(-abs(val_loss))*100))
        print(outputs.transpose(0,1)[0][0:3], bbox.transpose(0,1)[0][0:3])
        
#'''