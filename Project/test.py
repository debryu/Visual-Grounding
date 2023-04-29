# -- coding: utf-8 --
"""
Created on Thu Apr 20 10:24:49 2023

@author: Mateo-drr
"""

import gc
import time
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
torch.backends.cudnn.benchmark = True 
torch.set_num_threads(4)

start_execution = time.time()

#Bounding box data is bottom left x,y top right x,y 

path = 'C:/Users/debryu/Desktop/VS_CODE/HOME/Deep_Learning/picklesL/'
n_epochs = 25
init_lr = 0.0009
clipping_value = 1 #gradient clip
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
criterion = nn.MSELoss()
save_freq =1
batch_size = 8
resize_value = 515
resize = True

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
                #Check if the image is horizontal or vertical
                if(img.shape[1] > img.shape[2]):
                    orientation = 'h'
                else:
                    orientation = 'v'

                if(resize):
                    transform = tt.Resize((resize_value,resize_value), interpolation=tt.InterpolationMode.BICUBIC, antialias=True)
                    img = transform(img)
                #img = F.Resize(img, 224,interpolation=tt.InterpolationMode.BICUBIC)
                rsize = [data['img']['size'][0]/img.size()[1],
                         data['img']['size'][1]/img.size()[2]]
                bbox = [annotation[0]/rsize[0], annotation[1]/rsize[1],
                        annotation[2]/rsize[0], annotation[3]/rsize[1]]
        #decode = self.tokenizer.convert_ids_to_tokens(encoding_text['input_ids'].flatten())

        return {'img':img,
              'label':label,
              'bbox':torch.tensor(bbox),
              'orientation:': orientation}

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

import clip
import numpy as np

clipM, preprocess_old = clip.load("RN50")
clipM.to(device).eval()
input_resolution = clipM.visual.input_resolution
context_length = clipM.context_length
vocab_size = clipM.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clipM.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

def bwindowize(image,window_size):
        windows  = image.permute(0,3,2,1)
        #print(windows.shape)
        windows = windows.unfold(1,window_size,window_size)
        #print(windows.shape)
        x = windows.shape[1]
        #Also permute x and y torch.Size([2, 16, 16, 3, 32, 32])
        windows = windows.unfold(2,window_size,window_size)
        print(windows.shape)
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



transforms = torch.nn.Sequential(
                tt.Resize((224,224), antialias=True, interpolation=Image.BICUBIC),
                tt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                )
preprocess = torch.jit.script(transforms)

def bcomputeHeatmap(windows,window_size,x_size,batch_size,model,tok_labels,device):
       
        stride = 1
        kernel_size = 8
        patches = windows.unfold(1, kernel_size, stride)
        print(patches.shape)
        patches = windows.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        print(patches.shape)
        
        # Extract all possible 3x3 kernels with stride 1 from the 16x16 checkboard in dimension 0
        
        #From torch.Size([4, 11, 11, 3, 32, 32, 6, 6])
        #To torch.Size([4, 11, 6, 32, 11, 6, 32, 3])
        #Or to torch.Size([4, 11, 11, 6, 32, 6, 32, 3])
        patches = patches.permute(0,1,2,6,4,7,5,3)
        print(patches.shape)


        resulting_kernels = patches.reshape(batch_size, (x_size-kernel_size+1)**2, window_size*kernel_size, window_size*kernel_size,3)
        print(patches.shape)
        # The resulting tensor has shape torch.Size([4, 121, 192, 192, 3])
        print(resulting_kernels.shape)       

        '''
        ONLY USED FOR PLOTTING IN DEBUGGING
        '''
        plot = False
        if(plot):
            fig, ax = plt.subplots(11, 11, figsize=(40, 40))
            for batch in range(batch_size):
                i = 0
                j = 0
                for kernel in range((x_size-kernel_size+1)**2):
                        image_to_plot = resulting_kernels[batch,kernel].permute(0,1,2)
                        ax[i,j].imshow(image_to_plot)
                        ax[i,j].axis('off')
                        ax[i,j].set_aspect('equal')
                        j = j + 1
                        if(j == 11):
                                j = 0
                                i = i + 1
                        
                        fig.subplots_adjust(wspace=0.0,hspace=0.0)
                        

                plt.show()
                
        '''
        END OF DEBUGGING
        '''

        #First unify all the batches 
        resulting_kernels = resulting_kernels.reshape(batch_size*(x_size-kernel_size+1)**2, window_size*kernel_size, window_size*kernel_size,3)

        # Put the channel as the second dimension in order to apply the preprocessing
        # And also fed them to the model
        resulting_kernels = resulting_kernels.permute(0,3,1,2)
        #print(resulting_kernels.shape)

        # Compute the similarity score for each kernel of each image from the batch
        # 1)First we need to preprocess all the images 
        # torch.Size([484, 192, 192, 3])
        print('Starting preprocessing', resulting_kernels.shape)
        preprocess_input = preprocess(resulting_kernels)
        #print(preprocess_input.shape)
        print('Finished preprocessing with dimensions', preprocess_input.shape)
        
        # Reshape the tensor to the original shape
        preprocess_input = preprocess_input.reshape(batch_size, (x_size-kernel_size+1)**2,3,224,224).permute(0,1,3,4,2)
        print('last shape',preprocess_input.shape)
        print('last shape',preprocess_input[0].shape)


        '''
        ONLY USED FOR PLOTTING IN DEBUGGING
        '''
        plot = False
        if(plot):
            fig, ax = plt.subplots(11, 11, figsize=(40, 40))
            for batch in range(batch_size):
                i = 0
                j = 0
                for kernel in range((x_size-kernel_size+1)**2):
                        image_to_plot = preprocess_input[batch,kernel].permute(0,1,2)
                        ax[i,j].imshow(image_to_plot)
                        ax[i,j].axis('off')
                        ax[i,j].set_aspect('equal')
                        j = j + 1
                        if(j == 11):
                                j = 0
                                i = i + 1
                        
                        fig.subplots_adjust(wspace=0.0,hspace=0.0)
                        

                plt.show()
                
        '''
        END OF DEBUGGING
        '''

        # Move the number of channels before the height and width
        # CAUSE PYTORCH IS STUPID
        preprocess_input = preprocess_input.permute(0,1,4,2,3)

        '''
                test_image = preprocess_input[batch_index][60]
                test_image = test_image.unsqueeze(0).permute(0,3,1,2)
                print(test_image.shape)
                print(preprocess_input.shape)

        '''
        '''
        Similarity has to be a tensor of size [batch_size,16,16]
        Need to first reshape the 81 tensor to 9x9
        '''
        similarity = torch.zeros(batch_size,x_size,x_size).to('cpu')
        n_of_runs = torch.zeros(batch_size,x_size,x_size).to('cpu')

        # 2)Then we need to compute the similarity score for each image
        for batch_index in range(batch_size):
                
                text_input = clip.tokenize(tok_labels[batch_index]).to(device)
                logits_per_image, logits_per_text = clipM(preprocess_input[batch_index], text_input)
                print(batch_index,logits_per_image.shape)
                sim_score = logits_per_image.squeeze(1).reshape(kernel_size+1,kernel_size+1).to('cpu')

                gc.collect()
                torch.cuda.empty_cache()

                for i in range(kernel_size+1):
                        for j in range(kernel_size+1):
                                similarity[batch_index, i:i+kernel_size, j:j+kernel_size] += sim_score[i,j]
                                #similarity[i:i+kernel_size,j:j+kernel_size] += logits_per_image.squeeze(1).reshape(kernel_size+1,kernel_size+1)[i,j]
                                n_of_runs[batch_index, i:i+kernel_size, j:j+kernel_size] += 1

                
                gc.collect()
                torch.cuda.empty_cache()
                #similarity = torch.diagonal(logits_per_image)

        

        return n_of_runs


        unfold = nn.Unfold((256,256),stride=32)
        imagesC = unfold(windows)
        imagesC = imagesC.view(4,3,256,256,81).permute(0,1,4,3,2) #chops move downwards
        #imagesC1 = images[]
        transforms = torch.nn.Sequential(
                tt.Resize((224,224), antialias=True, interpolation=Image.BICUBIC),
                tt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                )
        scripted_transforms = torch.jit.script(transforms)
        
        imagesCP = []
        heatmap = torch.zeros(imagesC.size())[:,0,:,:,:]
        text_input = clip.tokenize(text).to(device)
        for i in range(0,imagesC.size()[2]):
                #clip returns the similarity of each image with respect to all the labels
                logits_per_image, logits_per_text = clipM(scripted_transforms(imagesC[:,:,i,:,:]), text_input)
                similarity = torch.diagonal(logits_per_image)
                heatmap[:,i,:,:] += similarity
                imagesCP.append()
        imagesCP = torch.stack(imagesCP).permute(1,2,0,3,4)
        print(imagesCP.shape)


        stride = 1
        kernel_size = 6
        #text = 'a pair of white shoes'
        #txt_embedding = text2embedding(text)
        text_input = clip.tokenize(text).to(device)
        print(text_input)
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
                        canvas_PIL = transforms.ToPILImage()(canvas)
                        
                        canvas_input = preprocess(canvas_PIL).unsqueeze(0).to(device)
                        
                        #Clip classification
                        logits_per_image, logits_per_text = model(canvas_input, text_input)
                        similarity = logits_per_image.to('cpu').item()
                                

                        heatmap[i:i+kernel_size,j:j+kernel_size] += similarity
                        number_of_uses[i:i+kernel_size,j:j+kernel_size] += 1
                        #print("kernel similarity:", similarity)  # prints: [[0.9927937  0.00421068 0.00299572]]

        heatmap = heatmap / number_of_uses
        return heatmap


#Batch computing
window_size = 32

from torchvision import transforms
toTensor = transforms.ToTensor()

#Do it for an entire batch
for batch in val_dl:
    '''
    1) Compute the heatmap

    2) Compute annotation embedding

    3) Feed the heatmap and annotation embedding to the model
    '''

    #Initialize a zero tensor
    
    images,label,bbox = batch['img'], batch['label'][0], batch['bbox']
    images = images.to(device)
    bbox = bbox.to(device)
    
    #print(images.shape)
    #asd = preprocess(images)
    #print(preprocess(images).shape)
    #plt.imshow(asd[0].permute(2,1,0).to('cpu'))

    tok_labels = clip.tokenize(label).to(device)
    print('labels',tok_labels.shape)
    #Chunking the images:
    #Dimension should be [batch_size,x_size,y_size,3,window_size,window_size]
    windowed_images,x_size,y_size = bwindowize(images, window_size)
    print(windowed_images.shape)
    #Compute the heatmap score for each kernel
    print(label)
    heatmaps_scores = bcomputeHeatmap(windowed_images, window_size, x_size,batch_size, clipM,label, device)
    print(heatmaps_scores[0])
    break
    
    
end_execution = time.time()
print(end_execution-start_execution)

