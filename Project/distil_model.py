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
import torchvision.ops as tvo

torch.backends.cudnn.benchmark = True 
torch.set_num_threads(4)

start_execution = time.time()

#Bounding box data is bottom left x,y top right x,y 

torch.autograd.set_detect_anomaly(True)

loadModel = False
path = 'C:/Users/debryu/Desktop/VS_CODE/HOME/Deep_Learning/picklesN/sim/'
save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/Deep_Learning/visual_grounding/visual-grounding/Project/distil-model/checkpoints/"
n_epochs = 250
init_lr = 0.1
clipping_value = 1 #gradient clip
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
criterion = nn.MSELoss()
save_freq = 1
batch_size = 8
resize_value = 512
resize = True
verbose = True
verbosissimo = False

def loadData(path,split):
    files =  os.listdir(path + split)
    dataset = []
    for file in files:
        file_path = os.path.join(path+split, file)
        
        dataset.append(file_path)
    return dataset
     

class BXfinder(nn.Module):
    def __init__(self):
        super(BXfinder, self).__init__()
        # Input 3 channels
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,stride=1,padding=0),
                                   nn.LeakyReLU(inplace=True)
                                  )
        
        # Shape [8 channels, 14x14 image]
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3,stride=1,padding=0),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.LeakyReLU(inplace=True)
                                  )
        # Shape [16 channels, 5x5 image]
        self.conv3 = nn.Sequential(nn.Conv2d(32, 124, kernel_size=5,stride=1,padding=0),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.LeakyReLU(inplace=True)
                                  )
        self.flat = nn.Flatten()

        self.bert_compressor = nn.Sequential(nn.Linear(768,256),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Linear(256,64),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Linear(64,4),
                                             nn.LeakyReLU(inplace=True),
                                            )
        
        self.lin1 = nn.Sequential(#nn.Flatten(),
                                 nn.Linear(124 + 4, 256), #3136
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(inplace=True)
                                 )
        
        self.lin2 = nn.Sequential(
                                  nn.Linear(256,128),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Linear(128,64),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Linear(64,4),
                                  #nn.LeakyReLU(inplace=True)
                                 )
        
                                          
    def forward(self, image, bert_emb):
        #x = self.down(x)
        #print(x.size())
        x = self.conv1(image)
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = self.conv3(x)
        x = self.flat(x)
        #print('nn tensor shape', x.shape)
        bert_comp_emb = self.bert_compressor(bert_emb)
        #print('bert tensor shape', bert_comp_emb.shape)
        x = torch.cat((x, bert_comp_emb), dim=1)
        #x = torch.cat((x, im_shape), dim=1)
        #print(x.size(), bertx.size())
        x = self.lin1(x)
        x = self.lin2(x)*resize_value
        return x


def starting_loss(outputs, bbox):
    #loss = 0
    #for i in range(0,4):
    #    loss += criterion(outputs.transpose(1,0)[i], bbox.transpose(1,0)[i])
    #return loss/4
    boxA,boxB = outputs, bbox
    boxA[:,2] = boxA[:,0]+boxA[:,2]
    boxA[:,3] = boxA[:,1]+boxA[:,3]
    boxB[:,2] = boxB[:,0]+boxB[:,2]
    boxB[:,3] = boxB[:,1]+boxB[:,3]
    loss = torch.sum(torch.pow(boxA - boxB,2))
    #print('boxA', boxA)
    #print('boxB', boxB)
    return loss

def final_loss(outputs, bbox):
    #loss = 0
    #for i in range(0,4):
    #    loss += criterion(outputs.transpose(1,0)[i], bbox.transpose(1,0)[i])
    #return loss/4
    boxA,boxB = outputs, bbox
    boxA[:,2] = boxA[:,0]+boxA[:,2]
    boxA[:,3] = boxA[:,1]+boxA[:,3]
    boxB[:,2] = boxB[:,0]+boxB[:,2]
    boxB[:,3] = boxB[:,1]+boxB[:,3]
    l1 = tvo.generalized_box_iou_loss(boxA,boxB,reduction='none')
    if(verbose):
        if(verbosissimo):
            print('boxA', boxA)
            print('boxB', boxB)
            print('loss1:', l1)
        else:
            print('boxA', boxA[0])
            print('boxB', boxB[0]) 
    return tvo.generalized_box_iou_loss(boxA,boxB,reduction='sum')


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

class simDataset(Dataset):

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
                
                img = data['img']
                bbox = data['bbox']
                bert_emb = data['be']
                sim = data['sim']
        #decode = self.tokenizer.convert_ids_to_tokens(encoding_text['input_ids'].flatten())

        return {'img':img,
              'label':label,
              'bbox':bbox,
              'sim': sim,
              'be': bert_emb
              }


#CREATE THE DATALOADER
def create_data_loader_CustomDataset(data, batch_size, eval=False):
    ds = CustomDataset(data=data)

    if not eval:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True), len(ds)

    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=False,  drop_last=True), len(ds)


#CREATE THE DATALOADER
def create_data_loader_simDataset(data, batch_size, eval=False):
    ds = simDataset(data=data)

    if not eval:
        return DataLoader(ds, batch_size=1, shuffle=True, drop_last=True), len(ds)

    else:
        return DataLoader(ds, batch_size=1, shuffle=False,  drop_last=True), len(ds)


#train_ds = loadData(path, 'train/')
#val_ds = loadData(path, 'val/')
test_ds = loadData(path, 'test/')



#train_dl, train_length = create_data_loader_CustomDataset(train_ds, batch_size, eval=False)
#val_dl, train_length = create_data_loader_CustomDataset(val_ds, batch_size, eval=True)
#test_dl, train_length = create_data_loader_CustomDataset(test_ds, batch_size, eval=True)


test_dl, train_length = create_data_loader_simDataset(test_ds, batch_size, eval=False)

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
        #print(windows.shape)
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
        #print(patches.shape)
        patches = windows.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        #print(patches.shape)
        
        # Extract all possible 3x3 kernels with stride 1 from the 16x16 checkboard in dimension 0
        
        #From torch.Size([4, 11, 11, 3, 32, 32, 6, 6])
        #To torch.Size([4, 11, 6, 32, 11, 6, 32, 3])
        #Or to torch.Size([4, 11, 11, 6, 32, 6, 32, 3])
        patches = patches.permute(0,1,2,6,4,7,5,3)
        #print(patches.shape)


        resulting_kernels = patches.reshape(batch_size, (x_size-kernel_size+1)**2, window_size*kernel_size, window_size*kernel_size,3)
        #print(patches.shape)
        # The resulting tensor has shape torch.Size([4, 121, 192, 192, 3])
        #print(resulting_kernels.shape)       

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
        #print('Starting preprocessing', resulting_kernels.shape)
        preprocess_input = preprocess(resulting_kernels)
        #print(preprocess_input.shape)
        print('Finished preprocessing with dimensions', preprocess_input.shape)
        
        # Reshape the tensor to the original shape
        preprocess_input = preprocess_input.reshape(batch_size, (x_size-kernel_size+1)**2,3,224,224).permute(0,1,3,4,2)
        #print('last shape',preprocess_input.shape)
        #print('last shape',preprocess_input[0].shape)


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
        similarity = torch.zeros(batch_size,x_size,x_size).to(device)
        n_of_runs = torch.zeros(batch_size,x_size,x_size).to(device)

        # 2)Then we need to compute the similarity score for each image
        for batch_index in range(batch_size):
                
                text_input = clip.tokenize(tok_labels[batch_index]).to(device)
                logits_per_image, logits_per_text = clipM(preprocess_input[batch_index], text_input)
                #print(batch_index,logits_per_image.shape)
                sim_score = logits_per_image.squeeze(1).reshape(kernel_size+1,kernel_size+1).to(device)

                gc.collect()
                torch.cuda.empty_cache()
                # Don't store anything for now
                with torch.no_grad():
                    for i in range(kernel_size+1):
                        for j in range(kernel_size+1):
                                similarity[batch_index, i:i+kernel_size, j:j+kernel_size] += sim_score[i,j]
                                #similarity[i:i+kernel_size,j:j+kernel_size] += logits_per_image.squeeze(1).reshape(kernel_size+1,kernel_size+1)[i,j]
                                n_of_runs[batch_index, i:i+kernel_size, j:j+kernel_size] += 1

                
                gc.collect()
                torch.cuda.empty_cache()
        return similarity/n_of_runs




#Batch computing
window_size = 32

from torchvision import transforms
toTensor = transforms.ToTensor()


''' 
 TRAINN


 



















































'''

from transformers import BertTokenizer, BertModel

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")

def computeBertEncoding(text, batch_size):
    #Encode label with bert
    bert = torch.zeros(batch_size,768)
    #print(text)
    for i in range(batch_size):
        encoded_input = bert_tokenizer(text[i], return_tensors='pt')
        output = bert_model(**encoded_input).last_hidden_state[0][0] #768 size
        bert[i] = output
    
    return bert


bxfinder = BXfinder();
bxfinder.to(device)
optimizer = torch.optim.AdamW(bxfinder.parameters(), lr=init_lr)  



import wandb
wandb.init(project="visual-grounding", entity="unitnais")

if(loadModel):
    bxfinder.load_state_dict(torch.load(save_path + 'epochs/saved/distil-v1.pth', map_location=torch.device(device)))


for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    val_loss= 0.0
    print(epoch)
    bxfinder.train()
    
    gc.collect()
    torch.cuda.empty_cache()

    i=0
    for bi, batch in tqdm(enumerate(test_dl), total=int(len(test_ds)/test_dl.batch_size)):
    
        images,label,bbox, bert_emb, similarity = batch['img'], batch['label'][0], batch['bbox'], batch['be'], batch['sim']
        images = images.to(device).squeeze(0)
        bbox = bbox.to(device).squeeze(0)
        bert_emb = bert_emb.to(device).squeeze(0)
        similarity = similarity.to(device).squeeze(0)
        # Compute the shape for each image in the batch
        #im_shapes = [img.shape[-2:] for img in images]
        #print('shape of the images ',im_shapes)
        
        #print(preprocess(images).shape)
        #plt.imshow(asd[0].permute(2,1,0).to('cpu'))
        #bert_embeddings = computeBertEncoding(label, batch_size).to(device)
        #print('bert embeddings shape', bert_embeddings.shape)

        #tok_labels = clip.tokenize(label).to(device)
        #print('shape of the labels with clip ',tok_labels.shape)
        #Chunking the images:
        #Dimension should be [batch_size,x_size,y_size,3,window_size,window_size]
        #windowed_images,x_size,y_size = bwindowize(images, window_size)
        #print('shape of the windowed image ', windowed_images.shape)
        #Compute the heatmap score for each kernel
        #print(label)
        #heatmaps_scores = bcomputeHeatmap(windowed_images, window_size, x_size,batch_size, clipM,label, device)

        #heatmaps_scores = heatmaps_scores.unsqueeze(1)
        #print('shape of the heatmap', heatmaps_scores.shape)   
        #print('similarity shape ', similarity.shape)
        #print('bert emb shape ', bert_emb.shape)
        outputs = bxfinder(similarity, bert_emb)
        loss = starting_loss(outputs, bbox)#final_loss(outputs, bbox)
        lossBB = final_loss(outputs, bbox)
        #print('loss ', loss)
        wandb.log({'loss': loss, 'lossBB': lossBB})
        #wandb.log({'outputs': outputs, 'bbox': bbox})
        #print(outputs)
        optimizer.zero_grad()
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(bxfinder.parameters(), clipping_value)
        optimizer.step()
        train_loss += loss.item()*images.size(0)
        #print('------   BB: \n', outputs, '\n ', bbox)
        #break
        '''
        if bi%100 == 0:
                torch.save(bxfinder.state_dict(), save_path + f'distil-model{bi}.pth')
                # Log model to Wandb
                wandb.save(f'distil-model{bi}.pth')
        '''
    train_loss = train_loss/len(test_dl)
    print('E: {} T Loss: {:.3f}'.format(epoch, train_loss) + " %" + "{:.3}".format(np.exp(-abs(train_loss))*100))
    print(outputs.transpose(0,1)[0][0:2], bbox.transpose(0,1)[0][0:2])
    if epoch%save_freq == 0:
        torch.save(bxfinder.state_dict(), save_path + 'epochs/epoch{0:05d}.pth'.format(epoch))
        wandb.save(f'distil-model{bi}.pth')


    ''' SKIP EVALUATION FOR NOW
    bxfinder.eval()
    with torch.no_grad():
        for bi, data in tqdm(enumerate(val_dl), total=int(len(val_ds)/val_dl.batch_size)):
        #for data in val_dl:
            images,label,bbox = data['img'], data['label'], data['bbox']
            images = images.to(device)
            bbox = bbox.to(device)
            
            recImgs = images
            bertLabels = data['blabel']
           
           
            latent = compress.encoder(recImgs) 
           
            outputs = bxfinder(latent, bertLabels.to(device))
            
            loss = loss = bb_intersection_over_union(outputs, bbox)#final_loss(outputs, bbox)
            val_loss += loss.item()*images.size(0)
            #break
        val_loss = val_loss/len(val_dl)
        print('E: {} V Loss: {:.3f}'.format(epoch, val_loss) + " %" + "{:.3}".format(np.exp(-abs(val_loss))*100))
        print(outputs.transpose(0,1)[0][0:2], bbox.transpose(0,1)[0][0:2])
    '''