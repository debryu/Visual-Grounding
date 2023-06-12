# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:13:05 2023

@author: Mateo-drr
"""

import pandas as pd
import os
import shutil
from PIL import Image
import pickle
import json

ds = 'D:/MachineLearning/datasets/refcocog/refcocog/'
out = ds+'idk/'#'E:/Users/Mateo/Documents/idk/'

obj = pd.read_pickle(r'D:/MachineLearning/datasets/refcocog/refcocog/annotations/refs(umd).p')

with open(ds+'annotations/instances.json', 'r') as file:
    jdat = json.load(file)

idk = obj

found = []
saved = []
failed = []
#print(obj[0])
#/filesPath =[]
i=0

files =  os.listdir(ds+'images/') #all files have train label

'''
for ref in obj:
    for file in files:
        if file[:-4] == ref['file_name'][:27]:
            file_path = os.path.join(ds+'images/', file)
            name = 'COCO_{}2014_{}.jpg'.format(ref['split'],file[-16:-4])
            shutil.copy2(file_path, out+name)
            
files =  os.listdir(ds+'idk/') #all files have train label            
for name in files:
    file_path = os.path.join(ds+'idk/', name)
    if name.find('train') != -1:
        shutil.copy2(file_path, ds+'train/'+name)
    elif name.find('val') != -1:
        shutil.copy2(file_path, ds+'val/'+name)
    elif name.find('test') != -1:
        shutil.copy2(file_path, ds+'test/'+name)
'''
            
labels = []  
dataset = [] 
split = 'val/'         
files =  os.listdir(ds+split) #all files have train label  
for file in files:
    for ref in obj:
        refname = 'COCO_{}2014_{:012d}.jpg'.format(ref['split'],ref['image_id'])
        if file == refname:
            labels.append(ref)
     
    anns = []
    for ann in jdat['annotations']:
        for ref2 in labels:
            if ann['image_id'] == ref2['image_id']:
                anns.append(ann)
            
    labann = []
    for label in labels:
        for ann in anns:
            if ann['id'] == label['ann_id']:
                labann.append({'label':label, 'annotation':ann}) 
            
    lb = []
    seen = set()
    for item in labann:
        item_json = json.dumps(item, sort_keys=True)
        if item_json not in seen:
            lb.append(item)
            seen.add(item_json)
            
    if len(lb) == 0:
        print('ERROR')
        break
            
    file_path = os.path.join(ds+split, file)
    im = Image.open(file_path)
    image = {
        'pixels': im.tobytes(),
        'size': im.size,
        'mode': im.mode,
    }
    im=0
    #f = {'img':image, 'metadata':labann}
    labels =[]
    for i in range(len(lb)):
        
        f = {'img': image, 'label': lb[i]['label'], 'annotation':lb[i]['annotation']}
        
        with open(ds +'final/'+split+ file[:-4] +'_lbl'+str(i)+ '.p', 'wb') as pkl:
            pickle.dump(f, pkl)
    
    image=0
    


    
'''    
for a in jdat['annotations']:
    if a['image_id'] == 131074:
        print(a)
'''    
'''    
maxx = [0,0] #x,y
for f in train_ds:
    if f['img']['size'][0] > maxx[0]:
        maxx[0] = f['img']['size'][0]
    if f['img']['size'][1] > maxx[1]:
        maxx[1] = f['img']['size'][1]
'''       
        
'''
with open('image.pkl', 'rb') as file:
    image = pickle.load(file)

im = Image.frombytes(image['mode'], image['size'], image['pixels'])
'''
'''
filt = [[],[],[]] #train,val,test = [],[],[]
for x in obj:
    if x['split'] == 'train':
        filt[0].append(x)
    elif x['split'] == 'val':
        filt[1].append(x)
    else:
        filt[2].append(x)
        
for i in range(0,3): #sort train first
    print(i)
    for ref in filt[i]:
        name = ref['image_id']#'COCO_{}2014_{:012d}.jpg'.format(ref['split'],ref['image_id'])   
        for file_name in files:
           fname = int(file_name[-16:-4].lstrip('0'))
           if name == fname:
               name = 'COCO_{}2014_{:012d}.jpg'.format(ref['split'],ref['image_id'])
               file_path = os.path.join(ds+'images/', file_name)
               shutil.copy2(file_path, out+name)
               files.remove(file_name)

'''
'''
for ref in obj:
    name = ref['image_id']#'COCO_{}2014_{:012d}.jpg'.format(ref['split'],ref['image_id'])
    print('searching', name)
    for file_name in files:
        #print(file_name[-16:-4])
        fname = int(file_name[-16:-4].lstrip('0'))
        if name == fname:
            print('found file', file_name)
            name = 'COCO_{}2014_{:012d}.jpg'.format(ref['split'],ref['image_id'])
            file_path = os.path.join(ds+'images/', file_name)
            if name.find('train') != -1:
                #idk[i]['split'] = 'train'
                saved.append(idk[i])
                shutil.copy2(file_path, out+name)
            elif name.find('val') != -1:
                shutil.copy2(file_path, out+name)
                #idk[i]['split'] = 'val'
                saved.append(idk[i])
            elif name.find('test') != -1:
                shutil.copy2(file_path, out+name)
                #idk[i]['split'] = 'test'
                saved.append(idk[i])
            else:
                print('FAIL')
                failed.append(idk[i])
            found.append(idk[i])
            break
            
        #print('checking name', file_path)
    i+=1
    
'''
'''
if name == file_name:
    newf = ds+'{}'.format(ref['split'])
    print(file_path, newf, i)#shutil.copy2(name, )
    i+=1

'''