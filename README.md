# **DEEP LEARNING PROJECT** on visual grounding






> Mateo Rodriguez, Nicola Debole




---
# Preliminary study
## Papers
We have been reading a lot of papers to understand how to face this visual-grounding problem, like for example:
*   [Generation and Comprehension of Unambiguous Object Descriptions](https://arxiv.org/pdf/1511.02283.pdf)
*   [Modeling Context in Referring Expressions](https://arxiv.org/pdf/1608.00272.pdf)
*   [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/pdf/2303.05499.pdf)

## Considerations
Of course model such as GoundingDINO are SOTA and would performs very well, we thought that the goal of this project is to learn how to build a model and so simply reading a paper and following the "instructions" is not the way we like to proceed. Instead, we decided to try and come up with our own solution (even if it is not the best) and using those papers just as a general guideline on what (not) to do.

## CLIP understanding
Since we had to work with CLIP it is important in our opinion to fully understand what that model is capable of. Our findings in no particular order:
1.   The model cannot understand negative sencences, for example even when prompted "not the black cat" the embeddings were most similar (using cosine similarity) to a cat than anything else.
2.   The model cannot understand spacial references or information about the location of the object. We tried with some pictures of a cat in many different areas of the image and then different promps with "left", "right", "up", "down" and other spacial clues but with no success.
That is why decided to include a **BERT** model in our pipeline, as it is capable of emedding spacial information in the text it receives as input.

## YOLO understanding (and why we choose not to use it)
Having already a box regression model (with also the ability to classify the object) is for sure a nice to have feature, but from our initial tests it was not able to recognize most of the object in an image. Sure, it correctly recognizes at least one object for every image but since in the REFCOCO dataset there are REALLY some strange images and labels, YOLO in not capable to find them all.
Also, for the purpose of the project it removes a big chunk of "freedom" (to try different techniques) so we decided not to use it.







# Code


## Disclaimer
Using Colab for a project like this is very annoying in our opinion, for reasons such as the limit on GPU usage and the difficulty when handling such big datasets.
Many time the kernel crashed for no reason at all and since we have available our own GPU we mainly worked locally. We then had to adapt the code for this ipynb file and manage to upload our dataset on the drive without having some errors in the process (due to internet connection or expiring sessions). We tried our best and hope it will be enough, but just in case we are also providing our GitHub repository were we developed the project and all the Weight And Biases logs for the different models we tried. These links be located at the end of the report.

## How to run the code

### Install the requirements


```python
!pip install transformers
!pip install git+https://github.com/openai/CLIP.git
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting transformers
      Downloading transformers-4.30.1-py3-none-any.whl (7.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.2/7.2 MB[0m [31m57.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.12.0)
    Collecting huggingface-hub<1.0,>=0.14.1 (from transformers)
      Downloading huggingface_hub-0.15.1-py3-none-any.whl (236 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m236.8/236.8 kB[0m [31m29.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.22.4)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (23.1)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2022.10.31)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.27.1)
    Collecting tokenizers!=0.11.3,<0.14,>=0.11.1 (from transformers)
      Downloading tokenizers-0.13.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.8/7.8 MB[0m [31m121.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting safetensors>=0.3.1 (from transformers)
      Downloading safetensors-0.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m77.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.65.0)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.14.1->transformers) (2023.4.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.14.1->transformers) (4.5.0)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (1.26.15)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2022.12.7)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.12)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.4)
    Installing collected packages: tokenizers, safetensors, huggingface-hub, transformers
    Successfully installed huggingface-hub-0.15.1 safetensors-0.3.1 tokenizers-0.13.3 transformers-4.30.1
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting git+https://github.com/openai/CLIP.git
      Cloning https://github.com/openai/CLIP.git to /tmp/pip-req-build-ztwbdek7
      Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git /tmp/pip-req-build-ztwbdek7
      Resolved https://github.com/openai/CLIP.git to commit a9b1bf5920416aaeaec965c25dd9e8f98c864f16
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Collecting ftfy (from clip==1.0)
      Downloading ftfy-6.1.1-py3-none-any.whl (53 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.1/53.1 kB[0m [31m3.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: regex in /usr/local/lib/python3.10/dist-packages (from clip==1.0) (2022.10.31)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from clip==1.0) (4.65.0)
    Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from clip==1.0) (2.0.1+cu118)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (from clip==1.0) (0.15.2+cu118)
    Requirement already satisfied: wcwidth>=0.2.5 in /usr/local/lib/python3.10/dist-packages (from ftfy->clip==1.0) (0.2.6)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch->clip==1.0) (3.12.0)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch->clip==1.0) (4.5.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch->clip==1.0) (1.11.1)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch->clip==1.0) (3.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->clip==1.0) (3.1.2)
    Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch->clip==1.0) (2.0.0)
    Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch->clip==1.0) (3.25.2)
    Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch->clip==1.0) (16.0.5)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision->clip==1.0) (1.22.4)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torchvision->clip==1.0) (2.27.1)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision->clip==1.0) (8.4.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->clip==1.0) (2.1.2)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision->clip==1.0) (1.26.15)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision->clip==1.0) (2022.12.7)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision->clip==1.0) (2.0.12)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision->clip==1.0) (3.4)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch->clip==1.0) (1.3.0)
    Building wheels for collected packages: clip
      Building wheel for clip (setup.py) ... [?25l[?25hdone
      Created wheel for clip: filename=clip-1.0-py3-none-any.whl size=1369370 sha256=5cf5c390f7c1949957909f66872a94c0a4425d6beeaabe1649d167c183a836cc
      Stored in directory: /tmp/pip-ephem-wheel-cache-i6c20yo7/wheels/da/2b/4c/d6691fa9597aac8bb85d2ac13b112deb897d5b50f5ad9a37e4
    Successfully built clip
    Installing collected packages: ftfy, clip
    Successfully installed clip-1.0 ftfy-6.1.1


### Mount the drive


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive


### Set the folder location
Import the "Submission" folder in your drive by adding a shortcut in your desired location (e.g. in your main drive folder). Set the location in the variable below:



```python
submission_folder_path = '/content/drive/MyDrive/Project/Submission/'

# Be sure to put "/" at the end of the path!
```


```python
submission_folder_path = '/content/drive/MyDrive/2s/Machine Learning/Project/Submission/'
```

---
# CLIP implementation
We decided not to finetune CLIP but instead freeze it and use the model to process the image and remove everything that is not relevant to the text in input. To be able to do that we are subdividing the image in smaller chunks and then for each "window" of chucks (could be 3x3, 4x4, etc.) we run the window thru CLIP and get the cosine similarity between the CLIP vector embedding we get from the text.

Then after we have a score for each window we can compute the score for each single chunk, normalize it and use this value as the similarity between that part of the image and the text.

Here we have an example of our idea to make it simpler to understand.


```python
import matplotlib.pyplot as plt
from PIL import Image
import torch
import clip
import os
from torchvision import transforms
toTensor = transforms.ToTensor()

demo_path = submission_folder_path + '/Report_assets/clip_pre_example/'
demo_image_name = os.listdir(demo_path)[0]
image_path = os.path.join(demo_path,demo_image_name)
print('Showing',demo_image_name)
image = Image.open(image_path)

# Convert the image to tensor
image = toTensor(image).to('cpu')
# Plot the image (move the channels at the last dimension)
plt.imshow(image.to('cpu').permute(1,2,0))
```

    Showing COCO_test2014_000000021750.jpg





    <matplotlib.image.AxesImage at 0x7fc88c7f7550>




    
![png](239782_239076_files/239782_239076_13_2.png)
    


Chunk the image in blocks of size 32x32


```python
window_size = 32

windowed_image  = image.unfold(0,3,3)
windowed_image.shape

windowed_image = windowed_image.unfold(1,window_size,window_size)
x = windowed_image.shape[1]



#################################################################
#             PLOT THE HORIZONTAL CHUNKING ONLY
#################################################################

#fig, ax = plt.subplots(x,1, figsize=(40, 26))
#
#for x in range(x):
#        ax[x].imshow(windows[0,x].to('cpu').permute(2,0,1))
#        ax[x].axis('off')
#
#fig.tight_layout()
#plt.show()


windowed_image = windowed_image.unfold(2,window_size,window_size)
print('Tensor shape:',windowed_image.shape)

y = windowed_image.shape[2]

fig, ax = plt.subplots(x, y, figsize=(2*y, 2*x))
for i in range(y):
        for j in range(x):
                ax[j,i].imshow(windowed_image[0,j,i].to('cpu').permute(1,2,0))
                ax[j,i].axis('off')

fig.tight_layout()
plt.show()
```

    Tensor shape: torch.Size([1, 13, 20, 3, 32, 32])



    
![png](239782_239076_files/239782_239076_15_1.png)
    


## Load CLIP
Load the model to evaluate the kernel (a N x N subset of the grid that will slide along all the image).


```python
import numpy as np
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50")
model.to(device).eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size
print("Device:", device)
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
```

    100%|███████████████████████████████████████| 244M/244M [00:04<00:00, 58.1MiB/s]


    Device: cuda
    Model parameters: 102,007,137
    Input resolution: 224
    Context length: 77
    Vocab size: 49408


## Run the CLIP convolution

First we have a function that computes the score of each chunk of the image. The score is based on how much is similar to the query (based on CLIP).


```python
def computeScores(windows,stride = 1,kernel_size = 6, text = 'a donut'):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  #txt_embedding = text2embedding(text)
  text_input = clip.tokenize([text]).to(device)

  heatmap = torch.zeros(windows.shape[1], windows.shape[2])
  #Count how many time each window is used as a kernel, in order to normalize the results
  # in the heatmap score
  number_of_uses = torch.ones(windows.shape[1], windows.shape[2])

  for i in range(0, windows.shape[1] - kernel_size + 1, stride):
          for j in range(0, windows.shape[2] - kernel_size + 1, stride):
                  #Initialize the canvas as a black image
                  canvas = torch.zeros(window_size*kernel_size, window_size*kernel_size, 3).to(device)
                  #print(canvas.shape)

                  kernel = windows[0,i:i+kernel_size,j:j+kernel_size].to(device)
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
                  with torch.no_grad():
                          logits_per_image, logits_per_text = model(canvas_input, text_input)
                          similarity = logits_per_image.to('cpu').item()


                  heatmap[i:i+kernel_size,j:j+kernel_size] += similarity
                  number_of_uses[i:i+kernel_size,j:j+kernel_size] += 1
                  #print("kernel similarity:", similarity)  # prints: [[0.9927937  0.00421068 0.00299572]]

  heatmap = heatmap / number_of_uses
  print(f'Result for the prompt:"{text}"')
  return heatmap
```

Then we create a function that plot the image based on the similarity score tensor.

It takes as input the similarity score tensor (referred as heatmap), the windowed image and the number of clipping.

The ``` num_clipping ``` parameter controls the number of time we **clip** the values. By clip (do not mistake with CLIP) we mean the mathematical __characteristic function__ (as the function that returns 0 if the value is less than 0 and returns the value itself otherwise).
Every time we clip, prior to clipping we subtract the mean. This is done because the values are very similar to each other so we compute the mean to get the average score and then "black out" chunks that are lower or equal to that averaged score.


```python
def visualizeCLIPconv(heatmap, windows, num_clipping = 3):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  augmented_scores = heatmap.clone().to('cpu')
  for i in range(num_clipping):
    augmented_scores = np.clip(augmented_scores-augmented_scores.mean(), 0, np.inf)
  augmented_scores = (augmented_scores - augmented_scores.min())/(augmented_scores.max() - augmented_scores.min())

  #Make a copy of the tensor
  hm = windows.clone().to('cpu')
  #Need to match this two
  #print(hm.shape)
  #print(heatmap.shape)

  hm = hm.squeeze(0)
  #print(hm.shape)
  hm = hm.permute(3,4,0,1,2)
  #print(hm.shape)
  #Put the number of channels at the end
  hm = hm.permute(0,1,4,2,3)
  #print('final',hm.shape)
  hm = hm*augmented_scores
  #Reorder
  #In order to print they have to be
  #print(windows.shape)
  hm = hm.permute(3,4,2,0,1).unsqueeze(0)

  #Print the results
  fig, ax = plt.subplots(x, y, figsize=(0.5*y, 0.5*x))
  for i in range(y):
          for j in range(x):
                  ax[j,i].imshow(hm[0,j,i].to('cpu').permute(1,2,0))
                  ax[j,i].axis('off')
                  ax[j,i].set_aspect('equal')

  fig.subplots_adjust(wspace=0.0,hspace=0.0)
  plt.show()

```


```python
CLIP_similarity_score_matrix = computeScores(windowed_image,
                                             stride = 1,
                                             kernel_size = 3,
                                             text = 'a donut')
visualizeCLIPconv(CLIP_similarity_score_matrix,windowed_image)
```

    Result for the prompt:"a donut"



    
![png](239782_239076_files/239782_239076_23_1.png)
    



```python
CLIP_similarity_score_matrix = computeScores(windowed_image,
                                             stride = 1,
                                             kernel_size = 3,
                                             text = 'a chair')
visualizeCLIPconv(CLIP_similarity_score_matrix,windowed_image)
```

    Result for the prompt:"a chair"



    
![png](239782_239076_files/239782_239076_24_1.png)
    



```python
CLIP_similarity_score_matrix = computeScores(windowed_image,
                                             stride = 1,
                                             kernel_size = 3,
                                             text = 'an old man')
visualizeCLIPconv(CLIP_similarity_score_matrix,windowed_image)
```

    Result for the prompt:"an old man"



    
![png](239782_239076_files/239782_239076_25_1.png)
    


In this last example we can see that having a smaller kernel size can increase the precision but cause the model to not capture the full object.

On the other hand if we increase the kernel size we could incur into the opposite problem (not very precise, as we get a very big "aura" around the object)

## Results
If we tweak the parameters correctly it is possible to achieve good results in blacking out all the non relevant part of the image.
Sadly, for the sake of this project was not feasible to have the model learn and optimize those parameters runtime (as the first part of the pipeline) because of the time required to compute a single step (around few seconds, that do not scall well for training the complete model since we have like >40000 CLIP steps to perform).
The complete list of parameters that can be changed to get better results at each step are:
```
window_size
stride
kernel_size
num_clipping
```

So we opted to just find the parameters that would perform good enough on average, run on the complete dataset and save the results as our final dataset. In this way while training the model we don't need to compute the CLIP step at all (as it is already stored in the dataset we made).


Anyway here are some example results that made us very optimistic about this pipeline:


```python
CLIP_similarity_score_matrix = computeScores(windowed_image,
                                             stride = 1,
                                             kernel_size = 3,
                                             text = 'a donut')
visualizeCLIPconv(CLIP_similarity_score_matrix,windowed_image,50)
##########################################################################
CLIP_similarity_score_matrix = computeScores(windowed_image,
                                             stride = 1,
                                             kernel_size = 3,
                                             text = 'an orange juice')
visualizeCLIPconv(CLIP_similarity_score_matrix,windowed_image)
##########################################################################
CLIP_similarity_score_matrix = computeScores(windowed_image,
                                             stride = 2,
                                             kernel_size = 6,
                                             text = 'the picture of a joung girl eating')
visualizeCLIPconv(CLIP_similarity_score_matrix,windowed_image)
##########################################################################
CLIP_similarity_score_matrix = computeScores(windowed_image,
                                             stride = 1,
                                             kernel_size = 2,
                                             text = 'a red purse')
visualizeCLIPconv(CLIP_similarity_score_matrix,windowed_image,20)
##########################################################################
```

    Result for the prompt:"a donut"



    
![png](239782_239076_files/239782_239076_28_1.png)
    


    Result for the prompt:"an orange juice"



    
![png](239782_239076_files/239782_239076_28_3.png)
    


    Result for the prompt:"the picture of a joung girl eating"



    
![png](239782_239076_files/239782_239076_28_5.png)
    


    Result for the prompt:"a red purse"



    
![png](239782_239076_files/239782_239076_28_7.png)
    


## Negative results
What happens if we look for an object that is not present in the image? Or that the CLIP model can't understand?

Lets show the results!



```python
CLIP_similarity_score_matrix = computeScores(windowed_image,
                                             stride = 1,
                                             kernel_size = 3,
                                             text = 'a trenitalia train that is not late')
visualizeCLIPconv(CLIP_similarity_score_matrix,windowed_image,3)
CLIP_similarity_score_matrix = computeScores(windowed_image,
                                             stride = 1,
                                             kernel_size = 3,
                                             text = 'a LLM that is not thought to be sentient by the general public')
visualizeCLIPconv(CLIP_similarity_score_matrix,windowed_image,3)
CLIP_similarity_score_matrix = computeScores(windowed_image,
                                             stride = 1,
                                             kernel_size = 3,
                                             text = 'airplane')
visualizeCLIPconv(CLIP_similarity_score_matrix,windowed_image,3)
```

    Result for the prompt:"a trenitalia train that is not late"



    
![png](239782_239076_files/239782_239076_30_1.png)
    


    Result for the prompt:"a LLM that is not thought to be sentient by the general public"



    
![png](239782_239076_files/239782_239076_30_3.png)
    


    Result for the prompt:"airplane"



    
![png](239782_239076_files/239782_239076_30_5.png)
    


As you can see even with "negative" or wrong prompts there are always active part of the image, this is due to the normalization step that rearranges the similarity values in the range [min_value, max_value].
This should not be a problem because during training the model is always prompted with description of existing objects within the image, but anyway it is worth considering this behaviour.


---
# Dataset

## Making the dataset


The dataset creation is a very long process that takes > 4 hours only for the training set, so we already prepared it in the drive with all the files needed at each step (from REFCOCO to our preprocess dataset). To avoid running the code by mistake (that would cause the dataset to be re-created from scratch taking a lot of time) we disabled the code.

You can inspect the code for the dataset creation down below, if you want to create it from scratch just set the variable
```
createDataset = True
```
and then run all the blocks.


```python
createDataset = False
```

To use the refcocog dataset we used the following code to place all the images with their corresponding labels:


```python
from tqdm import tqdm
import pandas as pd
import os
import shutil
from PIL import Image
import pickle
import json

if(createDataset):
  ds = submission_folder_path + 'Datasets/'
  out = submission_folder_path + 'Datasets/generated/'
  zip_path = submission_folder_path + 'Datasets/images.zip'
  extract_path = submission_folder_path + 'Datasets/imagesMixed/'
  obj = pd.read_pickle(ds + 'annotations/refs(umd).p')

  with open(ds + 'annotations/instances.json', 'r') as file:
      jdat = json.load(file)

  idk = obj

  found = []
  saved = []
  failed = []
  i=0

  '''
  #Drive file limit, so we need to unzip at each run:
  import zipfile
  try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
  except:
    print('Error with',zip_path)
  ##
  '''
```

The files all have the train name so we need to copy them into another folder with the correct names:


```python
if(createDataset):
  files =  os.listdir(ds+'imagesMixed/') #all files have train label

  for ref in obj:
      for file in files:
          if file[:-4] == ref['file_name'][:27]:
              file_path = os.path.join(ds+'imagesRenamed/', file)
              name = 'COCO_{}2014_{}.jpg'.format(ref['split'],file[-16:-4])
              shutil.copy2(file_path, ds+'imagesRenamed/'+name)
```

Finally we can select the split we want to make and we assign each photo all the labels and bounding boxes that it has


```python
if(createDataset):
  files =  os.listdir(ds+'imagesRenamed/')

  ################################################################################
  #                                  TRAIN
  ################################################################################
  labels = []
  dataset = []
  split = 'train/' #specify what split to make
  print('Creating ',split,' dataset')
  files = [s for s in files if s.find(split[:-1]) != -1] #ignore the data from other splits

  for file in tqdm(files):
      #Find the labels of the image
      for ref in obj:
          refname = 'COCO_{}2014_{:012d}.jpg'.format(ref['split'],ref['image_id'])
          if file == refname:
              labels.append(ref)
      #Find the bbox data
      anns = []
      for ann in jdat['annotations']:
          for ref2 in labels:
              if ann['image_id'] == ref2['image_id']:
                  anns.append(ann)
      #Remove possible dupplicates
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
      #Error check
      if len(lb) == 0:
          print('ERROR')
          break

      #Read the image data
      file_path = os.path.join(ds+'imagesRenamed/', file)
      im = Image.open(file_path)
      try:
        image = {
            'pixels': im.tobytes(),
            'size': im.size,
            'mode': im.mode,
        }
      except:
        print('no')
      im=0
      labels =[]

      #Save the new pickle file
      for i in range(len(lb)):
          f = {'img': image, 'label': lb[i]['label'], 'annotation':lb[i]['annotation']}
          #print(ds +'final/'+split+ file[:-4] +'_lbl'+str(i)+ '.p')
          with open(ds +'final/'+split+ file[:-4] +'_lbl'+str(i)+ '.p', 'wb') as pkl:
              pickle.dump(f, pkl)

      image=0

  ################################################################################
  #                                  TEST
  ################################################################################

  labels = []
  dataset = []
  split = 'test/' #specify what split to make
  print('Creating ',split,' dataset')
  files = [s for s in files if s.find(split[:-1]) != -1] #ignore the data from other splits

  for file in tqdm(files):
      #Find the labels of the image
      for ref in obj:
          refname = 'COCO_{}2014_{:012d}.jpg'.format(ref['split'],ref['image_id'])
          if file == refname:
              labels.append(ref)
      #Find the bbox data
      anns = []
      for ann in jdat['annotations']:
          for ref2 in labels:
              if ann['image_id'] == ref2['image_id']:
                  anns.append(ann)
      #Remove possible dupplicates
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
      #Error check
      if len(lb) == 0:
          print('ERROR')
          break

      #Read the image data
      file_path = os.path.join(ds+'imagesRenamed/', file)
      im = Image.open(file_path)
      try:
        image = {
            'pixels': im.tobytes(),
            'size': im.size,
            'mode': im.mode,
        }
      except:
        print('no')
      im=0
      labels =[]

      #Save the new pickle file
      for i in range(len(lb)):
          f = {'img': image, 'label': lb[i]['label'], 'annotation':lb[i]['annotation']}
          #print(ds +'final/'+split+ file[:-4] +'_lbl'+str(i)+ '.p')
          with open(ds +'final/'+split+ file[:-4] +'_lbl'+str(i)+ '.p', 'wb') as pkl:
              pickle.dump(f, pkl)

      image=0

  ################################################################################
  #                              VALITDATION
  ################################################################################

  labels = []
  dataset = []
  split = 'val/' #specify what split to make
  print('Creating ',split,' dataset')
  files = [s for s in files if s.find(split[:-1]) != -1] #ignore the data from other splits

  for file in tqdm(files):
      #Find the labels of the image
      for ref in obj:
          refname = 'COCO_{}2014_{:012d}.jpg'.format(ref['split'],ref['image_id'])
          if file == refname:
              labels.append(ref)
      #Find the bbox data
      anns = []
      for ann in jdat['annotations']:
          for ref2 in labels:
              if ann['image_id'] == ref2['image_id']:
                  anns.append(ann)
      #Remove possible dupplicates
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
      #Error check
      if len(lb) == 0:
          print('ERROR')
          break

      #Read the image data
      file_path = os.path.join(ds+'imagesRenamed/', file)
      im = Image.open(file_path)
      try:
        image = {
            'pixels': im.tobytes(),
            'size': im.size,
            'mode': im.mode,
        }
      except:
        print('no')
      im=0
      labels =[]

      #Save the new pickle file
      for i in range(len(lb)):
          f = {'img': image, 'label': lb[i]['label'], 'annotation':lb[i]['annotation']}
          #print(ds +'final/'+split+ file[:-4] +'_lbl'+str(i)+ '.p')
          with open(ds +'final/'+split+ file[:-4] +'_lbl'+str(i)+ '.p', 'wb') as pkl:
              pickle.dump(f, pkl)

      image=0
```

    100%|██████████| 515/515 [01:59<00:00,  4.31it/s]


After running the code above we end up with a pickle for each image mentioned in the ref(umd).p file, with the structue:


```
{'img': image, 'label': labels, 'annotation': annotations}
```

Where the labels contained all the sentences for the respective image, and the annotations contained the bounding box data.

Following this we tested a couple of variations of the dataset:

*   Have repeated images with different labels (complete dataset)
*   Have all images with a single label (almost half of the complete dataset)

And we found that **using the complete dataset didn't improve performance of our model**



## Clip Preprocessing

We used clip to preprocess the training data by applying a 16x16 moving similarity kernel, alongside using Bert to enconde the image labels


```python
#specify split to make
split = 'val/'
```


```python
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
import clip
from transformers import BertTokenizer, BertModel
import gc
from torchvision import transforms


if(createDataset):
  torch.backends.cudnn.benchmark = True
  path = ds+'final/'

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(device)
  batch_size = 1
  resize = 512
  window_size=32

  toTensor = transforms.ToTensor()

  stride = 1
  kernel_size =8
  plot = False

  def loadData(path,split):
      files =  os.listdir(path + split)
      dataset = []
      for file in files:
          file_path = os.path.join(path+split, file)

          dataset.append(file_path)
      return dataset

  class CustomDataset(Dataset):
      def __init__(self, data):
          self.data = data

      def __len__(self):
          return len(self.data)

      def __getitem__(self, idx):
          with open(self.data[idx], 'rb') as file:
              data = pickle.load(file)
              if True:
                  label = [data['label']['sentences'][0]['raw']] #take the first label raw
                  annotation = data['annotation']['bbox']
                  img = Image.frombytes(data['img']['mode'],
                                        data['img']['size'],
                                        data['img']['pixels'])
                  #turn the image into floats [0,1]
                  img = np.transpose(np.array(img, dtype=np.float32)/255)
                  #format as rgb b&w images
                  if img.shape[0] != 3:
                      img = np.repeat(img[np.newaxis, :,:],3,axis=0)
                  #to tensor and resize
                  img = torch.tensor(img)
                  transform = tt.Resize((resize,resize), interpolation=tt.InterpolationMode.BICUBIC, antialias=True)
                  img = transform(img)
                  #bounding box resize
                  rsize = [data['img']['size'][0]/img.size()[1],
                          data['img']['size'][1]/img.size()[2]]
                  bbox = [annotation[0]/rsize[0], annotation[1]/rsize[1],
                          annotation[2]/rsize[0], annotation[3]/rsize[1]]

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

  #load the file paths
  dataset = loadData(path, split)
  #specify only first label or all labels
  dataset = [f for f in dataset if f[-3] == '0']
  #load the data
  train_dl, train_length = create_data_loader_CustomDataset(dataset, batch_size, eval=True)

  #Load the models
  clipM, preprocess = clip.load("RN50", device=device)
  clipM.to(device).eval()
  input_resolution = clipM.visual.input_resolution
  context_length = clipM.context_length
  vocab_size = clipM.vocab_size
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  bertM = BertModel.from_pretrained("bert-base-uncased")
  bertM.to(device).eval()

  #chop the image into chunks
  def bwindowize(image,window_size):
          windows  = image.permute(0,3,2,1)
          windows = windows.unfold(1,window_size,window_size)
          x = windows.shape[1]
          #Also permute x and y torch.Size([2, 16, 16, 3, 32, 32])
          windows = windows.unfold(2,window_size,window_size)
          y = windows.shape[2]
          return windows,x,y

  #Clip's internal preprocessing
  transforms = torch.nn.Sequential(
                  tt.Resize((224,224), antialias=True, interpolation=Image.BICUBIC),
                  tt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                  )
  preprocess = torch.jit.script(transforms)

  def bcomputeHeatmap(windows,window_size,x_size,batch_size,model,tok_labels,device):
          patches = windows.unfold(1, kernel_size, stride)
          patches = windows.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
          # Extract all possible 3x3 kernels with stride 1 from the 16x16 checkboard in dimension 0
          #From torch.Size([4, 11, 11, 3, 32, 32, 6, 6])
          #To torch.Size([4, 11, 6, 32, 11, 6, 32, 3])
          #Or to torch.Size([4, 11, 11, 6, 32, 6, 32, 3])
          patches = patches.permute(0,1,2,6,4,7,5,3)
          resulting_kernels = patches.reshape(batch_size, (x_size-kernel_size+1)**2, window_size*kernel_size, window_size*kernel_size,3)
          # The resulting tensor has shape torch.Size([4, 121, 192, 192, 3])

          #First unify all the batches
          resulting_kernels = resulting_kernels.reshape(batch_size*(x_size-kernel_size+1)**2, window_size*kernel_size, window_size*kernel_size,3)
          # Put the channel as the second dimension in order to apply the preprocessing
          # And also fed them to the model
          resulting_kernels = resulting_kernels.permute(0,3,1,2)
          # Compute the similarity score for each kernel of each image from the batch
          # 1)First we need to preprocess all the images
          # torch.Size([484, 192, 192, 3])
          preprocess_input = preprocess(resulting_kernels)
          # Reshape the tensor to the original shape
          preprocess_input = preprocess_input.reshape(batch_size, (x_size-kernel_size+1)**2,3,224,224).permute(0,1,3,4,2)
          # Move the number of channels before the height and width
          preprocess_input = preprocess_input.permute(0,1,4,2,3)
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
                                  n_of_runs[batch_index, i:i+kernel_size, j:j+kernel_size] += 1


                  gc.collect()
                  torch.cuda.empty_cache()
          return similarity/n_of_runs

  def preprocessing(images,label):
      #Dimension should be [batch_size,x_size,y_size,3,window_size,window_size]
      windowed_images,x_size,y_size = bwindowize(images, window_size)
      windowed_images,x_size,y_size = bwindowize(images, window_size)
      #Compute the heatmap score for each kernel
      heatmaps_scores = bcomputeHeatmap(windowed_images, window_size, x_size,batch_size, clipM,label, device)
      #First, put every score one after the other
      heatmaps_scores = heatmaps_scores.reshape(batch_size,x_size**2)
      normalize = True
      if(normalize):
          tensor_mean = heatmaps_scores.mean(dim=1).unsqueeze(-1)
          #Choose to clip negative values basically removing data
          heatmaps_scores = (heatmaps_scores - tensor_mean).clamp(min= -tensor_mean.item()/10, max = float('inf'))
          tensor_min = heatmaps_scores.min(dim=1).values.clone().unsqueeze(-1)
          tensor_max = heatmaps_scores.max(dim=1).values.clone().unsqueeze(-1)
          heatmaps_scores = (heatmaps_scores - tensor_min)/(tensor_max - tensor_min)


      # Transform the tensor to apply the multiplication
      heatmaps_scores = heatmaps_scores.reshape(batch_size,x_size,x_size).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
      windowed_images = windowed_images*heatmaps_scores

      #Rebuild chunks into images
      recImgs = windowed_images.permute(0,3,2,5,1,4) #[1,16a,16b,3,32,32] -> [1,3,16b,32,16a,32]
      recImgs = recImgs.reshape(batch_size,3,512,512)

      #Encode label with bert
      bertLabels = torch.zeros(batch_size,768)
      for i in range(0,batch_size):
          encoded_input = tokenizer(label[i], return_tensors='pt').to(device)
          output = bertM(**encoded_input).last_hidden_state[0][0] #768 size
          bertLabels[i] = output
      gc.collect()

      return recImgs, bertLabels

  print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clipM.parameters()]):,}")
  print("Input resolution:", input_resolution)
  print("Context length:", context_length)
  print("Vocab size:", vocab_size)

  #Function to convert image into ycbcr, used for an experimental model, now it's irrelevant
  def rgb2ycbcr(image: torch.Tensor):
      r: torch.Tensor = image[..., 0,:,:]
      g: torch.Tensor = image[..., 1,:,:]
      b: torch.Tensor = image[..., 2,:,:]
      delta: float = 0.5
      y: torch.Tensor = 0.299*r+0.587*g+0.114*b
      cb: torch.Tensor = (b-y)*0.564+delta
      cr: torch.Tensor = (r-y)*0.713+delta
      return torch.stack([y,cb,cr], -3)

  def ycbcr2rgb(image: torch.Tensor):
      y: torch.Tensor = image[..., 0,:,:]
      cb: torch.Tensor = image[..., 1,:,:]
      cr: torch.Tensor = image[..., 2,:,:]
      delta: float = 0.5
      r: torch.Tensor = y + 1.403 * (cr - delta)
      g: torch.Tensor = y - 0.714 * (cr - delta) - 0.344 * (cb - delta)
      b: torch.Tensor = y + 1.773 * (cb - delta)
      return torch.stack([r,g,b], -3).clamp(0,1)


  for epoch in range(1, 2):
      print(len(dataset))
      i=0
      #replace train_dl with the others to create the other parts
      for bi, data in tqdm(enumerate(train_dl), total=int(len(dataset)/train_dl.batch_size)):

          images,label,bbox = data['img'], data['label'][0], data['bbox']
          images = images.to(device)
          bbox = bbox.to(device)
          #Run the preprocessing
          recImgs, bertLabels = preprocessing(images, label)
          #convert to ycbcr
          yim = rgb2ycbcr(images)
          yrecim = rgb2ycbcr(recImgs)
          #mix processed luminance with original chroma components
          mix = torch.cat((yrecim[:,0,:,:].unsqueeze(1),yim[:,1:,:,:]), 1).to('cpu')

          recImgs,bertLabels,label,bbox =recImgs.to('cpu'),bertLabels.to('cpu'),label,bbox.to('cpu')
          #To plot a processed image use: plt.imshow(recImgs.squeeze(0).permute(2,1,0).to('cpu'))

          with open(path+'processed/'+split+dataset[i].split("/")[-1], 'wb') as f:
              pickle.dump({'img': recImgs, 'Blabel':bertLabels, 'label':label, 'bbox':bbox, 'mix':mix}, f)
          i+=1
```

    cuda


    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel: ['cls.seq_relationship.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias', 'cls.seq_relationship.weight']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


    Model parameters: 102,007,137
    Input resolution: 224
    Context length: 77
    Vocab size: 49408
    515


    100%|██████████| 515/515 [25:00<00:00,  2.91s/it]


---
# Model

## BXfinder

Imports and configurations


```python
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

from PIL import Image
from transformers import BertTokenizer, BertModel
import time
import datetime

wb = False

path = submission_folder_path+'Datasets/final/processed/'
spath = submission_folder_path+'Datasets/model_checkpoints/epochs/'
n_epochs = 50
init_lr = 5e-5
clipping_value = 1 #gradient clip
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
save_freq =5
batch_size = 32
resize = 256
window_size=32
plot = False
```

    cuda


## Dataloader


```python
def loadData(path,split):
    files =  os.listdir(path + split)
    dataset = []
    for file in files:
        file_path = os.path.join(path+split, file)

        dataset.append(file_path)
    return dataset

class CustomDatasetPP(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx], 'rb') as file:
            data = pickle.load(file)
            if True:
                label = data['label'] #take the first label raw
                bbox = data['bbox']*resize/512
                #To use ycbcr images -> img = F.interpolate(data['mix'], (resize,resize), mode='bicubic', antialias=True)
                img = F.interpolate(data['img'], (resize,resize), mode='bicubic', antialias=True)
                blabel = data['Blabel']

                #normalize the images
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

#CREATE THE DATALOADER
def create_data_loader_CustomDataset(data, batch_size, eval=False):
    ds = CustomDatasetPP(data=data)

    if not eval:
        return DataLoader(ds, batch_size=batch_size, shuffle=True), len(ds)

    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=False), len(ds)
```

## The Model

The following model takes the initial part of the FASTERRCNN v2 network, which are 5 self-attention blocks that output 5 tensors at the intermediate attention levels:

![image.png](data:image/png;base64,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)

The model then takes this outputs and concatenates them together by upsampling until reaching the largest size which is [256, resize/4, resize/4], so concatenated the final number of channels is 320. To perform the upsampling we used PixelShuffle which has proven to give better results in retaining image quality while at the same time reducing the number of operations.
This tensor is then downsampled through a resnet reducing the number of channels to 9.


From here the model takes the BERT encodings from the label of the images and runs them through a fully connected layer that changes their size from 768 to resize/4 * resize/4 *3.
After this the labels are reshaped into a tensor with the shape [3, resize/4, resize/4].


Now we reshape the data into a bigger tensor of size [3, resize/2, resize/2] composed by the first 3 channels of the image tensor on the top left corner, the next 3 channels on the top right, the remaining 3 channels on the bottom left and the bert tensor in the bottom right part.


This tensor is sent again into the initial attention blocks of the RCNN model, which as explained before returns 5 tensors of different sizes, so we use the same upsampling from before just ignoring the 5th output because it's too small. Here we're using another jump connection that sums the 3rd output of the first attention blocks with the upsampled second output of this attention blocks.
After all the upsampling we end with a tensor size [320, resize/8, resize/8] which is again downsampled to 9 channels.

After this we flatten this tensor and run it through a fully connected network that gives us the 4 numbers of the coordinates of the bounding box.

![p1.png](data:image/png;base64,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)

![p2.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABQUAAAIKCAMAAACtL9lpAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAIuUExURf///+rq6hgYGAAAAIGBgd/f3xcXGERERPv7+8TExAoQGFmR0l+a3zdZgWSi6m2x/019tR0vRGip9FOIxGuu+5OTk6WlpVlZWbW1tSY9WS4uLvT09NLS0j5mk25ubmhrbrS5v//dps7U2y9MbnZ6fp6iqEZypRcXFxMTFL3CyBMfLiUmJzY4OsrQ1ygqLDQ3Ot/BkfvZo6WPa5eeqBgUD4FvU/TTnoqQmkQ6LLWcdVdbYFlNOdK2iOrKmMSpfw4OD5SbpS4nHZN/X25fR3R5gWFmbEFESISKkmtwd4WJjamutJCXoBscHkxPVEdJTFhbXnyCipKWm8XK0Xh4eKurq4eHh3x8fD4+PioqKk1NTZ2dnQ0NDTY2NnNzc35+fiwsLCQ4Pi9JUQwTFVeJlna5y4TQ5YfU6XvC1V+WpURrdRgmKm+uv016hmeis4HK3jpbZH7G2kx3g0dwezhYYUlJSQgICEMnDVczETMzM7JpI+eHLfyUMvGNL9yBK39KGRcNBPiRMaNfIGw/FcFxJpFVHC0aCc95KZeXl2pqajs7O3l5eR4eHkpKSlNTU5CQkKKiohoaLjMyWScmRA0NGEpJgXl30pOR/4B+319dpWhmtYaF6oyK9JCO+3BvxFRTkz8+bmTJbarhr+D04lTEXneu8Mfe+Pv8/pnC8+vz/BBx5arM9TuK6fT4/WOi7rnV90+W64HTiM3t0PHz8ff8+IqbsAAECSZ+54i48tTl+vz+/C+IJacAAAABYktHRACIBR1IAAAACXBIWXMAABibAAAYmwFJdYOUAAAAB3RJTUUH5wYMFRgQh0IiJAAAQzlJREFUeNrtnYt/E1eb3yU9UiQB4S7JSIrA5vKaAME2NoEYQwzJm5C8cd7Q3bbbbrf4hrEB0227uOFmc0u3d9syvl/avnvLu9vdXrfb/67nMjOakUbSjDSjkTy/7ycR0uiMNBbWl+ec55znBAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANBIgiGqgbDXlw0AAM4Q+aAWCRJFY15fOQAAOECcaM/efbb5EBoEAOwKuAT3H6iBg0ShoNdXDwAAdRImOnS4FgkeOHAEGgQAtDxcgkdrk+CBA3v30Adxr38CAAConViU6MOaJXjgwP49RNAgAKBlERKs3YFcg4eIEl7/GADsepLRVDjZFol4fR27Dj5N8GBdEjxw4OghTBwEwHWS2gS1VCqRjEcwIO8MXIJH6pQg0+Ax9veCGTMAuEqS0qmUcXUDdFg/8Q9oz966JcjAxEEA3CZJKf5HJpJNJot0GEqlkslsJOP1JbYgNU8TLOUjpkH8iwSAiygW1MhE4slEKkXQYR0kiA45JEFMHATAbYotqBE00WE6lUsmIxH00CpT1zRBEw3uoQ+yXv9MAOxeylpQIxhpS4ZTUTMden3xTUksR3TMQQli4iAA7lLdghoREx1ink0x9U8TNNdgm9c/GAC7FRsW1IhEkslcKm2iQ4xfBZkEP3JYgpg4CICb1GJBlZiZDv09z8aZaYJlNIjxWADcoB4LasRM5tkoOvRXYjn7Ae1xQ4IHMHEQANdwxIIaPp926OA0QXMN+uFDBKDROGtBjQrTDndvQNPmpgQxcRAAl3DJghrlph3uwpkfDk8TbKQGw7XtjVIjyPOA5sJtC2qYzbNpZuz+fDHXJXjgwD6XJg42VoLQIGgyGmZBDa7DVP3fJLex+VO5MU2wFFfmT3N/Hz/ROI4j3Q2ai8ZbUL4rtRsoftxoSq/H3g/UGAkeOHDY+YmD/NKPdzSS40h3g6YCFnTCgk6UVLWG4/OnGy9BaBA0GbCgAxZ0a660qQZPOlp4lUvwVIMl2NFxChoETQQsWL8F+TTBfQ2S4AFn508H00SnGy7Bjo7TRGnM+gFNAixYtwVdnSttxhnHFMKDWC8kKDSIyY+gSYAF67VgHTuv18oRhxTinQShQdBEwIJ1WrAB0wRLcWbH9sgvqPOERxLs6DjRSb9ASTbQDMCCdVkwliI62XAJOjNxkPXkO896JsGOjrOdKB4LmgJYsC4L5hozTdBEgx8T1VeG32sJQoOgWYAF67Jgik56IsEDB45+TMl6/gqYBM95K0GmwXPQINCTSVrB8dJKsGCdFjzjkQUPHKvLggmi8xc8lmBHx4XzRAkPfv9Ak2Jtaa3jyoIFfWnBcFNIUGoQtRWAQpbok4vV+KTesaBSLFlQC0XFo1g8mc0ox+Xs3QiLUTPqg7iFgNWmBbsk3WWe7qEeWNAWzSJBaBDoiKXp4qXqXKS0wwuPLFlQX2glKItjiWm76i9wkiKBiNq3SVH1+Q82Lai8e2/fZVjQAQuKIjLNIUGmQZSYAQpMC59asOCnVN+QuNkbW7JgRCIm2oaSwWyCQhlhx4h8EWFB+cANC17hoeDVK9QHC9ZvQS/qJ1QCtRWAIEP0mQUJXrr0GZGzCRKLFtTuhpTvXphy7HiK0vJFuAVTFOUP3LBgv/iju5e6+Z/XenoGxOOr17oGFAt2yUPtA1d7rvNGXT2X27uudsvz1RNgwUDzSRAaBJIU3fjckgU/v+FwNsOmBYNEyu+r6BEnE+KrKC2YzIkH7lmwvZ+62tuvDvLeMYv+LvfyO4PdzII3+9i9W6zFTdZvJrrOzXi1n93v6dadAAsGvCoiUxmUmAEiNfKFJQleuvSFwwkSmxZs4yFg4XgyFuKxqWLBWIh3k92z4GXqFTcD7V2DNMCMd5Xbro/9Tz3dA4N0uX2AS/Eab86O9V2+Psi0WTgBFvSuiExlUGIGWEyNuJIgsWhBmSKOs45wm/54kgk8pVmQOTLFZ6I5b8HBHkZfLx8XvE7X2vltX3ufGCYc4Ma7IsLA6+2XRQ/5CvNhjzDndRYCFk6ABT2tn1BFg6it4G8spkZcSZDYyhGnAgnDm/MHKYprFmQPsuKBhZ/YlgUlg6yn236LbnIjMsldJrpylWeNZXZE3HZf7em51csiwB5hvS7WTS6cAAs2qwShQd9jOTXiRoLEVo44yKK9qP44+x5mQqGYZsEgyQfV39V+jlhGfO2DihKZ1K7d4n926yzY1UuD/f3SgvxYN2umO8HvFvS2iExlUGLG31hOjbiRILGfHZEKjuXi0oLsFcKaBdmDhBsWFAbr5+OAMkOi0n3tCrNdwYJ9fFywXR8L9hlP8LUFva+fUAnUVvAz+tTIF19KxIEvfvnlV0KPn375y691XWZHEyR2Z8qkKCw0GOYTpuX3MEopzYKxNLmWHblMvZd5j5gr73JXe3cXH+8boEGdBQd5eniA6Jo2Lni1cILfLVhJgqdv375ty1nfVGmufzl9W/Y+5brk0KB/MaRGLiq9N3bk21/xO9+xg1+yx/KeCwkSuxbkI0u5bCItpjbI76GcLx3RP6j+rrXkiIXZuqi353IPF90VuiZ7w/pYsLfr6mAv9Q7IHDGPCgsn+NuCFYvIDNHQEPPQ8U4a0rU5+/15ovPfm511/pwUHA2ZNycybzs0pD+jSIMoMeNXTFIjX9PXly59Rxe//eIG/frSV/TJ55c+/4S+dSVBYteCgYxYQRcSi56Uy0joLMgr7rk2X3CQ94mv9xvmC/bqxwUH+KH+y+zpHrrep+RTtBPctCB7h2Psj/1njhiq7+87cubMPtNNST4+xJsfPLi/pLl8AfFypW33sTc6U4sFKxeRGeLSOk30/RB1fqMe/IG913kmNpP5hXfkxOsLnTqnnVKb/2C0YFHb8hZEiRm/YpIa+fzGry4x9/EI8duveAf4SxEQfu1AgqS0J23JghGD1zKRpJLNiyhjhJFIjN3oHlTDbjUFZa6fnAjTfrmHBYHiuFwTcrnrsnZ7/So7MtDFgr+u9oGryvnaCS5a8BCT3b6PuW8Pafbaf0yG9h+VVqPeTx+JSqmkK1C4/5Bsfuao0YL6tkf3HazJglXqJwgLfk9necT2vXLsBHUePysDxJKUymmZav6eCk4zNucWvHPBrG0FC6K2gk8xSY18R19x63126XPxzFf0S3Hw1w4kSFgMVyTC3VpZq6dyD7gmC2biJf/2FCx4TIjpzOG9h7StiQ8zb32478gx0sd1Cgdpr2x+jN1RZUd7WPNDsrnuHGPbfWUtmA2Hyn3g1YrIcAteINFz7STt2HE1yDt3gc6LO8fF8Tsdx+mC8NvtgtPOqSHjcX6MWfB2Z+epC+KxoW0lC0KDvsRk1Yi03pf0xRc3bnzHOsuf/5K+++I7Hh/WnyARndlcXBes7V4L2qqwYMmCEaJom1GEBgvuFebaq0V3x5QtSQ7voSOHz/Adi4+c4R3gg+yck8SsJ5uf0Zp/KJrv38Nlx548fPKIOGBsa27BGFMglYtgqxeR4RY8K/00pEZ+RN+oUR51nOcqOy88yW86uRTvsNuC00g97xseBxKdG7p9XnSOi9pWtCBKzPgQs1UjcgTwIt345de/JGLh4Oc8T/IrY8RYY4JEnf5cECEsaMuCVCRCYyyoRG4faqGgslf7h3TssGiwh/VtDxzlfWatvWZBFgru15rzLvYZ2kN7Dh8obmtiwVg8Vyh9ZvJbVr1+gugRn+M3F9T+753C0B47dJsp8Q6dZ2HgCRbznRVxH/el5jTWIS748GyHGB28wyPI4raVLYjaCv7DJDXyqcwGX1Qjwkuf3rjx2Vef3bjxqbFVTQkSXdXqnPwy71YL2sSGBXlxxYS6zKHEgvv37VG7uPtoj2avPQc+Ji66PYeky/bRwQOq8vapzT9WDh3h94hOHj76EW9W1LbYggUFlrGglSIywoKsK3vixPdqTHdCdIKVZ09wl52mb+h0x232/A+8zQ90u0NvQc1u57nxmC/l6GBx22oWhAZ9htmqEWUAUGZDPmUu/Ez0mb8oavkZ0d/7HdtvaCzfz6MaWLAGCxZEWGzBY0Qf7ysOCrnSDnzEDHaQztBRdm+/+F/LfEjOaM2PsubslKPceKx3XdTWYMFMW9TwV1r6K/a7f/8fWCgiIyx4lhmQOs8Jf/FhQlI70Ty46xzqON7Zce54x1CnbH5H9HQ1p53VmosTZSDJb4vbVrUgSsz4i5zo8Rq5cUOZLfOZYsGLIlz8lD4xJpKJ/uHvWdqqpCLR34UFzS1YnXQ4W2LBj459TB/vV0O6kzqt7WUR3clD+1mgeIiFiIeUMPHIHm1L94Na88PCgofUly1qW7DgPypSoDn/mKj6hOgh2fu9c/ubC1p2RFtyLAb6jlPHue87jp/r4Dlk/v/3dPz27dt07nbxMOJpOS6oWrC4bXULsnBT1q0EfiBO+syvYjc5UPitsB7vEYvZg+z2S0O7XxP9/j+p14EsoEEsWLMF+Qf4e6XjgmdUm+0T4Vyhi3vywJ6P2H9HWcx3VIn7zui2dC90oI8UBg7ZbXHbggX/IBey8O/c71tZNyctyIPAO5qljqtzZk7xQ6fpNP0gbk8zLf7AT1G4rb7EcfW84zoLlrStbkEsIvEXqeKsh5IhFj3jX339HY8VPycxLqibNX1JZExqsZfh69HG+3Q1WjBbdVQymazQuoIFB7q6LKz71VFlbVy7/uX0bdn7FEoOmliw4odnGgse3ic09TEdViPAvep4HhPjyT372OOTh/ay2yNiNs3RD/Vbuh/Wmp/kzTULFrc19IhlZriA2d+EleXDSnbkVMcJkSI+wYO2O0TnTl8Qkdkd3s/tZP3iO8Q7zKdIez3hNLX5+dPCmZ13dBYsblvdgpCgzwhSUYinzJGW+RGi777ih/i9i8YJNV8S1VKHSKdAJdFZsGAkkU4nxPybYEq+diaRSodSCbMJ2lHZY4mkEkUnhNLKCfpvpLF1oKIF+UIPXji/v19fJb+751Z//y1eObqEK73Snv03zZvr30rflvRlZhwZFzwjzHVYy3J8SMf2KVNf9okeL4sNz4jbD4UoP9JLUDTfrzTfr7Ngcdvi7IhBhKZ/6xZKyQgLnmDak2OIt0WA9w3TUed5UlaTnBdJ4HM8Z6IsidOcJpuf1jfXLFjctqoFUVzGdyTIGOMZ4r2vPi+5p/BtjSuN5FclpZvroVmQT4aLKqvf5G2cfbvSaaJQW8nrxOTUVl47QZ4rTmhTT4gHDBYsah2oYkEepxENDlKhGMy1XqJe/n/pVOhuWT61e1DnNENz3VsVta3ZgmVyxIf30Id7jxzjsdsx7r2je4gOHdn7kZgfw2O9Q6KbfExZEnfg4z1793GU5odF8yMfkWiuWbC4rclMGfYPWCULWigrKHvEp08NnRO6PN0purkXbg+dOzd0W1kCMiT8OPSDuiROnniq0PyOrvmQnHs4VNK2mgVRaNB/xEI2ykwXuEihmpJopJ8qKNAsKErBxCkXCKfbKBzN8Tg1x0LDrNnC4LjssoTFNvHhaJIS6TCXhHpC0GBBQ2vlXStbsJ+v/LiqlYfm9VSvd4sFwSV186/xMjJiwxHNaQPs/vX27utXiC4bBFfUtjYLFhQYKM4RsyiOr/84qmjtwNGT4ow9cjHJx9JudFBJ9u5VpVrcfO8BnQVL2prPmg5KEZb5e6+qwSEyPLxgrxarreaVLQgJ+hGTBEl1fl3rwEmRAgMFC2aSIrhk36NsOC0W2qXUpUxhimZEcdWEOJKjDDvEo8ksJfjZvEvGx8iianwa5kfZK8VD6baYeKxrrbxrRQt23+ozHOxXfdhH/d2i53u1n68Rvtl/mR26LILHmwWnac1v8WPsVa4NXrkuTze0rcGCBgUGSmfK7N93RElhyPkyR/cd3KvWS9gv6iTs33f0wGGxhOTwPgUuuZLmsgbDvv0lbcuuoAsmo2VXP1fbbqTIgqfJ1nbFP3TaaFzRgtiAxJ+UJkiqUmNqxBRjdiTJN1diPVceaWqL9LJMCeL7FRXzF0LsJs2lGAulY+LsWCiU1p8Q5w7hHWwWn/DetLG1fJ/KsaDgqthVTj57XQ3mqH2QN7giCk/3Dra3Dw6KPm5vd8FpWhWt6/xlWeA32NcrKrQWtbVrwWCi5AtqsnZESfOWFlCowME91tvuq1BNoXyNjSpbz/Ekrh3v1QxRJQtipqBPKU2QVKXG1IgpOgtGkrlQiIksF8qEcnxGt/r7GKIg31IpRlEW1AVZxBcUUV+OHRNnp0LBEO9Bq2v5Y7zkDckK/KlAcWv5rtUseLmnp1cdBWQd4m6t3cBN6mrvpkF2aIDFfJfppujjXis4bUB7sW7eJSbeYoAbtbitXQuaULDgx2eO6FR15Ih1rdlqfvjMh7VV1qq8gsR2ldVaqVBlFatGfEylBIkptaZGTNFZMKnkNdrigXiS77KuPpGiCK8eGKes2GkpG2jjUV+WX4VolGRH2opOCEhTc5sUtVbeqooFe3SpkC59Z7frGvXwAT7msqvK/+xhn85pxubtciyRv0FxW0ctaFY4xgVqri/YhLuxQ4JAxXaCpNbUiCk6C2ayyVRhu2FjLBgIpQLhUCAdZt97HtfxyxbVpgv96aBmDiUW5HfJvLWlWPCKOryniwV7mdKYvG72tvfebL/FDvP/u3v5biM6C/YWXvay8tL8tritkxZsPLYr7levLOMpqCbja2wmSGpOjZhiHBfMhDT1BbS3iSgqS+cCuXSAe5Kfk6NcMpmkdDJSOEE3kKizoElrS+OCBflpA31dUmbtg7fab11p72W93N5+nvTo6erqoiuizKq++TU5Lqi+QXFbn1mwepVBTyWIyoL+xlaCxMnUSKBgwUhKfKdyhVkxWo5YpHbjFKc21rnltxGe80ipczeSJSeEhSlVC5q1rmLBgZuiSvQtdcKglvS9yZMirGNLV9ltF7vt4gW0+tUXV4ppXaGbavN+nQVL2vrOgk2sQUjQ79hKkDiZGgkULCi32IzpVrGzIyke3MV5h5h1kKPsjyClWbNE4Qr0RubzBblD28RLaRY0a13NgmK3uW6eAhZ19FkI2MeFeI3vpsRixCusXzzAbi+399CAXHSnxHdacz4WeJ33nwsWLG7rQwtW2X3EWwlixxF/YyNB4mhqJKDrEbN+eTRHpFsnItaOpNIyZRKIihRwiGsyWijsbohL2/ic4lRIv3aET7IxaV2tR9xHdKuvl8drslrqVfbKV/qukJjv0j4oBv56aVBbEqeN8snmN/nqOLW5ZsHitn60YOWd6DwDu88BOwkSR1MjAf24YJwJL2fcZinMOrJRuf9woE0sAk6k2tQFcYKUQcnBXJQFkPKEVEremrauOi7Y0y/Wf2g1owduDRIN3hpQOrq3RH/5Znu3NqXQYMH2rv5erblqwZK2vrRgU27NjvoJIGAjQeJsaiTQhLWm+4ue6VU3kmvvbi+hW02IaAd6r5drXdLWnxZsQg1CgkBgMUHicGok0PwWvEbF7qrIdW1OjQX8aUErJWYaCorIAInFBInDqZFAU1qwv1/3RLXqgUVcHrDctL/fpxa0UGKmkaB+AlBJMA1Wx/lEWtNZ8Ga/wYLuwd6nUJHQVxZsKg1CgkAjZqF4OsPZ1EigCS3oDf6yYNUSMw2VIIrIAJVsygo17cVeCVjQjxasVmKmYaCIDGgCYEFfWrBJaiugfgJoBmDBOi34Uf0+q41DdVmwKTQICYKmABas04Jkr5KgUxw9ZFjnUwOel5hBERnQJMCCdVkwGCVPokEuwbqLD3hbWwH1E0CzAAvWZUHRsfywfqvZZP/HVF9/WOKlBiFB0DTAgvVZMBBjneKTtjYZcUCCexxacuZdiRkUkQHNAyxYpwVFRHWooRo8soc+cGjKlFclZlBEBjQRsGDdFuQR1aH99cvNsgSdXG3hTW0F1E8AzQQsWL8FuUr2NEyDHzm72sILDUKCoKmABR2woNDg3sZI8EOn59g1vsQMisiA5gIWdMKCoj5BIyYOHj1JlHJ4jl2jayugfgJoMpIUsrSA2VnSlkpHeIrNz5Gr5KD7EnRimqDptQ81DkgQNBlJr23TrNj9IBsxcXD/IXemlwQb+68SisiA5iKS9IbdNzguNOjqjBnHpgmWXntD/+6xag6A3YrLEwf3uSZBAABwBlc1eATjaQCApqfNvYmDByFBAEAL4Nr8acenCQIAgCtEPnBj4uBRSBAA0Cq4MX/anWmCAADgCkHHJw5CggCAlsLp+dN8mmB9xfUBAKChxHJExxybMePeXGkAAHALBycOHiH6AAVYAACtRpLoY0dmzGCuNACgNXFo4uBHRFFIEADQisQ/cKDwKqYJAgBal/onDh49RpSDBAEArQqv2nemHglimiAAoLWpb+KgQzuvAwCAd9SzYzumCQIAdgM1Txzcu4c+gAQBAK0P1+Bh+xLENEEAwG4hji2KAAD+pjYNYpogAGDXkImUg9mu3FOQIADAB9Sw7TEAAOwiIEEAgL+BBQEA/gYWBAD4G1gQAOBvYEEAgL+BBQEA/gYWBAD4G1gQAOBvYEEAgL+BBQEA/gYWBAD4G1gQAOBvYEEAgL+BBQEA/gYWBAD4G1gQAOBvYEEAgL+BBQEA/gYWBAD4G1gQAOBvYEEAgL+BBQEA/gYWBAD4lSAVSHt9MQAA0HhCBQsmvL4WAABoPOGCBYNeXwsAADSeLDrEAAB/E0KHGADga8LoEAMAfE0WHWIAgL9RusRhr68DAAC8ISctmPX6OgAAwBviQoIhry8DAAA8IoYOMQDA3+TQIQYA+Jo4OsQAAF8TQ4cYAOBvWJc47vU1AACAd7QRxby+BgAA8I4M5by+BAAAsEAskk26wj/9A1deNh6JZLz+zAAAu4VYPJymFiSUSsKEAIC6iaiVX+4OtxAjykVHkXsBANRFRESBw6Nj4/dajYn7k9yF0YjXnyEAoHWJJZhGpu63ngFVHgyzH6DN648RANCqBFkgODXmtcnqY2wK87IBADUS/AXRw9aNAxXG70KDAICaCIZopMUDQanBSWgQAFADsTSNTHhtMGeYRN0aAIB9okQPvNaXU9ylENbpAQDs0Ub0yGt5OcYE+sQAAJvEQjTstbscZJQIy0gAAHaIE017rS4nGcHO7wAAW0TpsdficpSHKGkNALBDZhelRgTTqOYKALBDnEa89pbDDKOQIQDABuFdlRvhjFLa6w8VANBCRGnUa205zBiyxAAAG+y2YUEGEUpsAQAsQ7QbVhAbuEtJrz9VAEDrsAstOAwLAgCsswst+BiL6AAA1tmFFhyllNefKgCgdYAFAQD+BhYEAPgbWBAA4G9gQQCAv4EFAQD+BhYEAPgbWBAA4G9gQQCAv4EFAQD+xoYFHw0rWN+yTjllUqlbM/5ocuru5CO+z8nD4UlxaGL4IX/wUHkgX3p6WN0FYIw3m3jIzrIua1gQAGADGxYcpakKFhwze6FRusva3yWaGucS5HeUB8NKSa8xXuV1WLmKMbXY4V213tck3b/3iGh4ikbGYUEAgAvYsmClgqz3zS0oDk7f5VseT4zQMJPb+CTdHefiE15TLTg1rrfgI5KR4jjR+L0pvkneKNMhLAgAcJ4aLfiIiesx99L05Mjw8OgEFxkRezg6RVO82TA9GCbVgvce8I3uJlW1DU9Os+dH6WHBgg/Fa2sWHB+hcanWSfaS/OCE9ZrYsCAAwAa1WXCMqW5yhCZ4GavH9x+ySO7RFE2O8mr3D0dHuN6GaYr5TbWgiO1EQKgyTONT/EnFguzBhM6C9x7Ktnf5UeUVYEEAgBvUZsHRqTEe4E2yLusUV9QkDwbHeLjH5MWMxnu8j+9pPeIHYpyPSDe0xwLFMbrLYzxhQfZiw3oLTvPn2JN31fZ3MS4IAHCFWrIjE2p4JgQ2Oa1ojb3QlLDWJLs7LIK5UbpLgsl7Unp6C7I4kklPsSB7cF9nQXZoQosI7+nvwYIAAEepzYLTj0e423j4xu5MjU5LC06QwiMhNjVHTKJfO14cC94bHxmZ1iw4PTIyrrPgI96tHlECwPHHNiQICwIA7FBbj3iKRsbG7outjMcfMSNOSQvy4ULBmGbBMRHIPZYnKSobH5fiu8dfQbUge/BYZ0FmQGnCe2qKGRYEALhBTRacFtHdfXVD9/G7Ivgb4+GePtZTLTg+Jab/TZIyF3p4RLXkMD3QLHjvLumz0A/pwWMSne2JkZGJezaABQEANqjRgsxLLEIbmXhw9/49OWo3zE0nwr2xx5PjBgveeyCmA07L+YLTk3yAUD4/QSMFC/L+dMGC03RXseykrUgQFgQA2KK2HjETIHPZMI38s7v0eHRyZEpMnhm+94BJ7bHIhRgseO+x6Ns+GtGvHVFes2BBJlP9dJhhkvYbpxE5HAkLAgBcwNY6Yi0mm2DimxwfGx6emJjkRuSvMck9Nf6IRXAiPByWp8i+7PjwMO/cTovGyjpi+cRjuY5YfaCL+saUxcRj6vJlWBAA4AKoKQMA8DewIADA38CCAAB/AwsCAPwNLAgA8DewIADA38CCAAB/AwsCAPyNExacHmOUX+qrzXe2PvEZFgQANAonLDgqy2k9LvNKWomFQq0FWBAA0Cw4Y8HJ0dGHRCOwIACg5XDGgvw1HozI6gcT9x/J6tP3JsbGxlX5jY+N8Ttjo+rbPZJLhnlP+tED9dj9Mct19WFBAIAjOGfBcbG5yLhahFqUiKGRMWlBUYKQHefPTYid69hzD5ktaWqC78PJmo0/5KferV+DsCAAwAaOWZB5bkSo7+H4mNiD5CENP3jMX55ZUEiQ3RkZZc4TpbSGx1irSX5s6v7EQ37vId2dGH+sVmKFBQEAjcHB7Mhd3hF+NDoud6e7J6U4/IBbUG5FTKLk9F3itar5ebwsoSgiOM7b3hUbs0/RNCwIAGggzliQ77E0xWK5e3zazP3RSRbvTaj1+JnolP3Y5S5MwzQxJp8bZsYjYb0pbkaxZ8lU/dcDCwIAbODcuCDfPknGhcO8Vn5hJyVlJ041RzxKD8bktkrsnnJM7E6sAAsCABqJgxa8N0LT48QFx3cqntbHgnITORn3PaaxaTUWHNNb0Kk51bAgAMAGDlpwmkaUrdd5j/jeFB8XHBse5aKT28jJ9+LuE88xa04ULDgh700gRwwAaCjOWPD+2NjYI57znSZ6NM2nvDziuxBPT6vh3n0+BYbo7tj4Q7558UOanJ4WrtQsyP4X7UfqvRpYEABgBwdzxMxsYuc4psBJoonxSWVAUIjuMbtHI3dJzg2cEM89ntBbUB6zt/UwLAgAqBcnLDjGk7uPxuQUlwcP70/cGx/j60emR+/zY6M8TTI+Ojo++mj80aSyy9z06KhoL56890jcjt9/+ABrRwAAjQWVtQAA/gYWBAD4G1gQAOBvYEEAgL+BBQEA/gYWBAD4G1gQAOBvYEEAgL+BBQEA/gYWBAD4G1gQAOBvYEEAgL+BBQEA/gYWBAD4m11owUnKef2pAgBah11owWFKev2pAgBaB1gQAOBv0tqGmbsGWBAAYIOc3Cp4N0EU9/pTBQC0Dkm+JdyuYoIo6PWnCgBoHSJyq/RdxCMirz9UAEArEdptA4OTmC4IALBDeJd1iceJ2rz+TAEArUSQ6IHX5nKS+0Qxrz9TAEBLkaZhr83lIOMjWDkCALBHnOi+1+5yjodEGa8/UQBAi5GikV2TJn5ElPD68wQAtBqZX+yaPjGTYBSjggAAu7QRDY97LTAnmIAEAQA1ESa6uws0OEoUwrIRAEAtJJkGJ7yWWJ1M3IUEAQA1EyeiyVbOkYxNsp8gh+4wAKBWsiFmkcePWlOEY/en2NWns15/iACAViaW5B6kqeHRsvzhP/8X3vCHf1j+okYnh/l1U7oNgSAAoD5i8RS1KOlwxOtPDwCwO4i0JVJm/Ms/ErZ5MuMFT8R7/9G/SpmTjGO1CADAXWI5bsAfnz7ziucvXrIrCHv9OQAA/EqYOdA7BSoiZB5E4UAAgCckiF567EDOS1QOBAB4QqY5JPjs2QyFMAIIAGg8YZqd81qAguezCAYBAI0nRvTKa/8pvMbIIACg8USImiMUfPbsDarpAwAaT5zIa/upzGHDdQBA40nSjNf205ihpNcfBwDAdzSTBV9j5jQAoOE0kwVfIT0CANCTTFXEmd4jLAgAaFYy1UqtODLHGBYEADQrYeq8XYFOZwbRYEEAQJPCQsHbHRW47Uww6KYFX7yQ/8OCAIAaYKHghUoWvOBMMFjWgm9mlDozM29rtSCfimhnOiIsCAAoEPtF5VCQB4O/sLnUIpvMpVKpXFK/g0dZC74iJYqrPVjkBnxlY30eLAgAKJCsEgqKYNBOmjiS0yVWcpHC+7hrQTvAggAAjViITnVU4RSFLAeDGeHAaC5F6VxUeFAZVLRmwZmZZy/evXzOHs29mnn96hW/9+blu3dveZu3M09fz8789Oz563dv5kQP+ifZlltwZkY5MvOGv9jzl09ePJt5CQsCAKqQJLpTzYJ3yHIwGA/xbdwy/HWZaDJtaaJQXHkjKxYkmpmdEdVnZujJqxl+6AW9e8kOPeeHZt+x/1/Nvp6lt/q26rggPZl595roDX9Rej37svxbwoIAAAkLBY9Xk2BHx3GrwSBzaqhNuSdF0xZSFGrZgnMsGpx99pye8OdmXjx7zZ9/I8X4mgWENPv82Rx/lujlHHekzoJcji94MdfZd0+fzc3AggCAalgJBa0Hg+zVoloHWBFNJipPtmpB/mCGRX7v6Eddoxeiu8yf/JGr8NksFdrO6Sz4nFeMmX32VLzZG1gQAFCF8qHgiRqCwSyToNpMs2AgxjSYtWDBOWlBPs73ln7iRQBnX/MhvrmXYg9NYUHhw1fP5F1ZsPAlO12z4LtncpRQhI6KC2FBAEB5yoSCJ77vpM7vT9gMBplSNQnqLMg1yBxawYJyjssbacE5cYjp7/krJj/W8X1Hs6/e/GhqwWeKMTULzigW/FE0eg4LAgAqUyYUvHCu8/jt7+mczWAwTKGg9kBnwUAwROHyFnwqRgB5uauX+h4x5/ksvXhKsh9sZsGSHrFqwRei24weMQCgCoZQ8MKJs/yPsycu3BbzqIfoG1vBYMbQRG9B/j6Z8ivo3hGf9/JmlgeAIuU792T22YuZn2RQ+JTezfFEB7stseCPrO07fXZEtSDrGz9/NvcaFgQAVMQYCl7oJKbBs3TuwtkTd8QswR9sBYMJQwuDBdkbJcpb8Eei2ZezYuSPGezdzKtZFhXOzc6+/enV7DsuySevZp+w27kSC87KtiYWfE2zb99hpgwAoDJFo4In6HxHx3k6oUV/+ierB4NpShheWy+aBKUrVFN4+vYdM9qrOWGwOaY+nhR5OsNTIqzP+9MTotfP2e2L0uzIK3r30zP9fEHVgiyWfPLiGSwIAKhI8ajgKbp9W1lJMnSOOn8wPMmCwYovFiSK6B4aLRgh+tcVF8gpBRWEwZ6rB18om9Y9Nz/H0LaYOU2YsCAAoAzxkgTx+c7O8/Iei8POny6eM1hx47Yskf6h0YIBot+1tEzYzorgSm2f0Bs+Llhm709YEADASZckiL8hLSNy55tOKp4zmK70akXaK3qYoj9uqAXnntCsGDKEBQEA5WCh4IkiCw4RDWkPfiiS5Amif/NvKxXmD+k3K0kbH4YsWnDGRl2Zim3nXr2deVNuG3hYEADAyFFnkQRv06lTfJLMOaHHb+h749Od9O/+fbUtSsrzH1BxHwDQXESIjCN/fJLMhXN0tuMUnT/bcWeoKFQ8TZT7nfIvl6R0UkfK+DBtMRaEBQEAjSNlWB2iTJIR02WOi+itKBQ8V1kdSeOwYdG4ICwIAGg+ioLBs7fFo9O3z/Kb74+f7SgOBSNVXk0/rdpowRjRf4QFAQDNRnEwWJEqoSAXnX4mjdGCcaL/VJMFLW4sN/fUWjtYEACgp2RksALVQkGebcnpHhktyJ6rYSfOF69niV7/VLXdj6/5GjylCCEsCACwjo1gsFooGAi0GTxpsCDTbVsNFnxHr1+9ljUDK8qS3r398Qn9KCsUwoIAAOtYDwarh4I8A2IoI6N7kKK05V3Z516I3u3TF3NvxKzn1/Sm8KRuSdxTsb6Ot37CK2yJWoJkcUdjWBAAoGA5GKweCorBv8LIoN6C4gnLFnzHpfaUZueevpB1p3WV93UWFBUHmSNfPXsh+syzM4anYUEAgBUipetHTDlhIRQMBKK6VjoLsjeJBixbkHVwnyhbjHCez5KuYoJeczP0k24kUNZVtQgsCABQSemWzFVgyJI2YmkKZZX7BQtmQ5SO2bAgL8H/o9K3nXlCs6r4nr548YJeshtlXdzc7Ozzd7OqIl/OPrf48rAgAEBH1lIweELsn1SdYIhIbsRZsGAbyUL8NrIjTH3vpOqI6Ik6LDijrsVTZ8/8RLNab/mlYcc6WBAAYJm0lWBwqHI5mQLBNFFa9IoVC0b4gaA8YNmCT0nLiDx/MUs/mceCfFBQyR/PPdGnUGBBAIAN4haCwRNVSgvqiKVYsBZNBIUFg4koe5SSS0psWHCGlA2ZOG/0A36G9McL9uLCkE/fzVafVAgLAgDMsRAMWg4FOZGooZRMVM2XWLfgj/Ra7M4pMySGqdB6C/Jk8uzsnEig2Fs6AgsCAHRUDwZthIKCbDitKDAdLgwnWrbgc6Y2Zrinz97Sk6fPnr8mXZynt+Bb9uBHHii+tdcdhgUBAEaqBoO2QkFJMJJNZiNB/SHLFnzCtfeC77z5Uqi0zGk/iV7zDP347F1R1gQWBADYolowaDcULINVCz5/JbK9b1495TevX5br7IoGvPXcKwXrU2VgQQCAnirBYA2hoBk1rCN2DVgQAKCHBYNDFXAmFIQFAQDNS7rihiHOhIKwIACgeYkkK2JhBbEFYEEAgL9pJgu+hQUBAA2nmSw4Q0mvPw4AgO9IVi8a3TCewIIAgIaTJbvr3NzDUrVEAABwlpCt4ldu8gIWBAB4QFhXKMZbXjs0+QcAAOyQJbJX/8otnjo0DxwAAOwRpdm5+h1WN89nKeT1RwEA8CVBonfeJ0iePsGoIADAI+JEs6+8DQef8mpd6A8DADwi+wvmwZev3rzwhp/e8lKEaUsbSQEAgBtkwuQx6WTM6w8BAOBrMm25VMgrBfJtoQAAoJVhKvP6EgAAwENgQQCAz4EFAQD+BhYEAPgbWBAA4G9gQQCAv4EFAQD+BhYEAPgbWBAA4G9gQQCAv4EFAQD+BhYEAPgbWBAA4G9gQQCAv4EFAQD+BhYEAPgbWBAA4G9gQQCAv4EFAQD+BhYEAPgbWBAA4G9gQQCAv4EFAQD+BhYEAPgbWBAA4G9gQQCAv4EFAQD+BhYEAPgbWBAA4G9gQQCAv4EFAQD+BhYEAPiVJBVIen0xAADQcII6Cwa9vhgAAGg8aU2Caa8vBQAAPCChWTDh9aUAAIAHRDQLRry+FAAA8IKQIsGQ1xcCAACeEFYsGPb6QgAAwBOyigWzXl8IAAB4QwgdYgCArwmjQwwA8DVZdIgBAP5GWNDriwAAAM/IMQnmvL4I4A6hVCqRjEcwGRSASsSZBeNeXwRwh8Iy8XQqlYQPATAlxr4iMa8vArgDC/NTqRAZiKZyyWQ2guoZAGjk0CHetSgrI2ORSFsykUqRiQ8jkYzXVwmA18TRId61lKwPz0QiyWS4xIepVBg+BD4mhg7xrqVClYxMJJtM5lJRow5FOqUtEsGvBNgdZFg3yAp/bKlVkg8m1fbl4N83R6l3kN/qJ2OZeJOGUZZqBQWlD9MmPkQ6BbQywUTRb7VDhO1+3bPhUP3vWkooWWu4kmmL1v/2JqSaUBg2K6ZFIvFkMlXsQ6SXQUsSlF/0+QVHUb7tdvSTlQpcdPQ6FqUHayoMG8uJk/POfjJ58aLRpsu71lw3UkmnIL0MWhY+B3B+6f2y46ys5u182YVyFtfWHb+OjZUlqmn1M7dyfun9pvOfzNY80QfN5of6q+dWSi9nm3QcAAAhwbwLCpSssSjM6i9/mDnQeQUqImQeTNn9ZIIfEK269cm8zzddyW4nLwjpZdBCZJl7nA92NLbyVu2TIFpy7zqWmQbb7H0ymRDl3bIyYz1P6eZKrrqjZfP0MiG9DJqHtKsSFNGgpRmGGXcluLy8YD0qlYRdlSDrFjfbLlZuB6fl08uIDYGXRIhWXJUPs4+l1SbMOa7aeHkjby8YzLjYHZZsN1m92sZ10YvTy002NAB8Ro4W3f2qL29ZWnkcc905yzv2RgYTbmt5eb3Jvv5eXA5PpzTZxwB8B9GWy/LZtNQlZjGpy86xqGONFG27fEHL883VJfbKRrAg8BbXO8Q8BrPQJY4TuX0d1nSskXY9OGVd4rTXf//G3wVYEPiQBoRgy6tWeqJJWnD7OpYXKGnjo2nAvw8rzVW3u4UsGLO0xCiEDDSwQMT9EGz5vZXveiMsuGNr5nQDLLhO1EzZ0RayYNiKBLFPGLBEIyxoKeJphAUtBaUaDbDgsq3vv/OLmUtoFQuyX9vPvqrKZ+hqAyvAgmWBBRuF/feN0q8+v1SVz39FUU9+INBawIJlaUILtisU7jmA/mVbxIJtRL+uLsFLl35te70Q8COwYFlgwUZh931jIbpoRYKXLl1EggRUBxYsCyxY9KuiIh7FsslsTDku/8ywP2Pqg6DhlYKVlwvbtWCY6FtrFvwWCRJQneoW3Hy/vbCzVo8Q6rDgyurCwuqKxbk8G2s7O2sb4pq3dhZWS9cAO21BW1fX8hbUBg8DWkXKdFAcDyt/gxH+6yTngaf0f+WRKj+m3equRF9ak+ClS18iQQKqUtWC63ki9l89lQ5qtuDmInvvvNWaN+xt5lljpsENcVp+w10L2ru63WBBLRYMhiiUiMTDor+pJlUUC5Komqi3YCztrAWtpUaQIAFWqWZBJsGFLWaYxTqWk9VqQaaZ+bXN5fUda53lBXrPV2Ms8XmBa1x5JeJ21IKFq9vxiwW1uyFl+nmYrwqilFwBIy2YktbRWzDBDjpoQaupESRIgEWqWXBBsd96foeFPBvb89yJy+sLa8tb+Z337I54WvnDYQtuK37ZXFxkYd3m2vziFg+72HutsMvYXF6QZyh/bCxw621Snt3bEfdKXtBRC1q7us2FigZvTQsG1QXZMdEjToaFFKUFkzlhHZ0FIxROOmhB66kRJEiANapZkEg3vLZNtDDPi+6t0PzC/AJ/Li8W4G3z4MtxCy7oX3WNvfeCKP1A8zv5HWLP7fDgb3nLEPOtaZHZemn06qgFF/RlKCxe3S6xYJthYTglmZgymgVjIV7GsWDBWDoUc9KCFVIjX32FBAmogSoWXKf5woMNml/hKlnlXlvlX/1t5j/ugnnaqPAitVqQdK+6medOWuGOI1raFPekYaRt5AWurea1R9ulEnPUgravrvUtKIcFg0xE+k4mCwTj/INVLCgdmdD+ynOUDThowQqpkS+JkCABNVDFgmv6QS9lpI3pakWctc7urfMG65WHxmq04Iq+8qHyYDGvhqfsCjbzed7xndddYMFcK/nSIMxJC5pd3XzFq2t9C0pSTHH6qhT8QYrimgXZA+495dksTxo7aMHyqZFPydyCSJCAKlSx4Ir+W7wqe6jzxA4vKD5kjzaVgNBpC27qy928lwbeYWaSV8xvl9ijLV3Hd2V1e14ph8X0XZq6ddKC5le3bri6dcPVtb4F1ViwzaAVLr4MyW6vsGCQ0jHVgrFQNOakBSukRj658YmpBZEgAVWoYsFNfb9vVcpukTZ0FlxjB3kM5LwFmWALcn0vdbJksCA/uEPrxvfKy3Zmw3GO9ohNrm7H5OoqDRU4Y8GuHoWu2uxYU3ZElsKJ5eLSguyvL6xZkD1IqBaM6GcZln9lk/f9z//lv5YerJAa+ZJ+fdHcgkiQgMpUy47MqxmKxQXmvm3lG75ZsOAG7axXmUxYqwU1lS3NryjvuGjwzPL8/KbWMV1ZFSNwC6zF5qJ5gVRHLWjv6ly0YI/qmZ4iC3b1qzcOW5D1eeV+SXKmjOgep8V0GGnBWJpySuNMkpNijqz4yibv+5vf/KZUhOVTI58yPZazIBIkoCLVLPie8qt8IsgSH/rL8wnCW7yTXLAg084SVd6qrVYLblB+ib/yGp+YvCCnQ+c39Z7ZpkJ2ekVsEyIG4tbKaNlRC1q7uirbGThjwatdgstFFuwh9cZpC0ZCRLlsMkq8uystGBGTpyOFB8a/XPs94t8IjCIsnxr5/Fc3vi1rQSRIQEWqrh1ZYr/QOztyhcQO5bcXiGc9dRbcoioZADML/smf/lnREbO1I2vsvReW8mJDTHZ/e0mkpnWeWSfd6NwS0xJrvMZjsu1VTnFvtIIF//TP/6L4ULW1I/auzrYFSz+ishbsMhwfuNrXd40fH6Se6+KGH+zr6+nmB6+3d/f1XK7TgsoKulCOR4TaDOqCBQM5pyxoFGH51MhnfLiwrAWRIAGVqL6O+P0C/7bL9bnvd2heuEVnwU2qOFnQ3II///xz0bfcdB3xOn/vRWmz9SUl+NJ5hvlOl53enmeNtzaFfQTFFqtgwd/+/PNf/vavDIeqriO2d3V2LVj6EVmz4ADRYC8N8gdE/eKmvesKO0i9A+xg/9XeXqJrNVgwYiyQkE0qjyPKGGFEVFPQHugbZ2qopvAbHYoIy6dGvqJfXqpgQSRIQCUs1ZTRdXg3qre2aEHGf/vrgnfK1ZTZKERTm9UX7Fa+vMoWZPyN7oIs1ZSxd3WlVLZgkQitWbCfWLh3i3uun7cSN33Ux9v18xb9A+1X+UPbFnQPot9UhYmwQmrkIn1y8eLFG3Txs7INkCAB5fCostbPKpp3PK6s9duSC/K6spb2EWkiLGvBPpkjluOCXdyJ1+iW3oK9V3jDQepuF2FgN/W2nAV/89//x/8k+q6M5D67eLGyBb9rrh1eQFPhtQWFd/iQXLNYsCDCZrGgJsJqOWJlpsz1nr7+Kzzu0yw4QIP9jF61hfH85rBgycEiBf4v8TtCX12qQPke8Vdka/tB4C8aZMGfK/OXf/4XjbHgzxb5G6adxljQ8hX99V9VyxF3Kz3i3v6+PoMFr0oL9rPOcJNasOK4oFRgQMzA+aQ2C35CaXSIQTmaw4LsS/6/m8qCLPz6s+ayILui/2NlXHCA6LoyBqhZsIv6jWe2kAU1BXKyRF9XsOAX5VYYf02U9egLBlqAhlnQ+LYlcU6DesTF16HxW6NweA+0URYsc0U/l1yRpezIdboi3HeF33TLmwHRvvvqQKtZ0KBATopuWC6wWuDzG7YmigK/4f244J/IUbjmGRdUsxFNMy74t8oVWbLgZaK+6/297La9j252qTdXrl/vlTnilrFgiQIDfL1y2QRJBZAaARXx2IKFiSBNYkHdzJTmsODf/t+/K1yQBQu23ySi3oE+KUTlpvvqIPvjSncNFoxrWYWkyT0TkvbTEKafwv8zf/UqCRKkRoB9TCy4WWnS3eZ6+UMrC+a19MpY8G8Nk4LNVtBtlN5btnJobcF87l41C/7NX+tnKZez4IqtQ5sLleaUV7GgToGBsha83NWt3hWzZETPt72b3R3oUm/YU9cvay2U23IWjKfi4v2CqYS+XCqZ3DPBynLJklMsx6DVEiRIjQD7lFhwczVPtKD4ZCNvNMHWAilLJMwObfP1xOullQzMLGj8fgdMLLi9SLQoyjes8W2Otss8aXZoni/pe5+3t3bEMGOaY2rB9SX+8Rgku7KTp/zOSplDW3wp8Wa5rVMqWbDkI2pUZa2kEjpF+MdV2FKzKSxYLUGC1AiwT4kFt4m282oplEWjCXgBgdVFQ6kC/aFFXsggT1Ys+HclF1JswTVa5NUCt3hBh/ltdu+96ZNmhzZom3vH1gq6v/irkkOmFszTztq8oVLM5jz7BHYMa6n1h5ZokwvZvgX/zuSCPLCg/u1L75ngsgVtJ0iQGgHVKLbgBi1uMIXsbEhr8FpRGyviwcoK+0ZvCRFsmBzipy7xpcVbVixYSpEFN/O8Qoyo4SzK+Yu61mLr382V9cKTJod4+PWe3dtetWNBE8wsKAsJLtAKe0v+mL3/qvh5F+i9ySF2O8+uaGF+3b4FzS7IAwsm+EcWSYWSUm+Fe5lkmqK868z6zfF0OhnTrpLdtKUoFGYRWEp84LFUjrVPhNT2cXY4E46mwspJtj4FuwkSpEZANYotuK1zGPvqcpOsi5JVa7zQ/nsZIK6bHFL6fmvMP45YcIUriO8otyzu8coNq6I4wQ6tFZ40OcQfbjIXLbthwSVxbI1WN/Oi+0/zm+viX4Rtdgmlh5Q9oLY2l1vWgnxcMEwUTqfJcC9OoWRC7MhJlEqze23aVfIn08kw36guJzqjcQoH2nh7Il6VNR2iRCBE4WRO2cXJXm7aXoIEqRFQlWIL8jBne02Oey3mN4RJVnkPU8RaMhyaNz0k+n7CeU5YUJWO2vsWNe15dPVea7dU5tByflFxnvMW3BHHVsTWK4uyrKpgg9fkNjnEhCl78q1lwZSojRpWLBijkNhiTn8vkAsFeVyW5pcV5uWno9pV8idFs7ZAVnguR8FAytg+Ip5IhGPyFDsWtJcgQWoEVKXUgtt5ZWiPbzMiTcLUqOyltr4wT/PvzQ8p8nHMgjzvoqZplhe4cTfyef6f/kmTQ+vKBbhhQdndX+LXyqLmVWVXEb496dqy6aEd5eNtLQsWNlriFpTjg+w3RXcvID/KHDuDiO9KHCLtKpU7YmO6NAsIY8yQMRn2pVhb0T7GgsdY4QezNU/RToIEqRFQnWILCpGs8C2VxL6S0iTMM0qlvBVec9X8kLr/r2MWXOXJZ2nBTTn8yINOJbbSniw9pNrPDQtuEi29X8qLa13MK95f5mUG5ZYsJYfUn6q1LBgW2yy1aRYMy7c33FOIKNpL6S0YzKUVifKAMM76wdni9m38+XgN44IBOwkSpEaABUpjwS0R1GxtzvPoSzHJWmETpvUtWUC55NCaIg0He8QrS/Lg5qKaDl4spGfVJ0sOLeRV5zlvQRFw0pL4Ed+TlrfeUKbllBxaUSvQtpYFjeOC5hZMi15zMmNmQaJoJJLkp2dYHJhj8WCcosb2gUxbimoaF7STIEFqBFig1ILr4tu8vUbzCwusX7e4IDYd1+0oucS/2aWHFkh1nnPjgpsyP5zPr2ivVFCTfLLk0KZa4NkVCzK9rShbki7op7/IrU5KDm2rW7K0sgVjWj84pusbpwuXxW/1FsyK4cOEaJyjoBwHjBrbc4Ih6SjbK/esJkiQGgFWKLbgmnDYNr3fWljQLLhDa3xrt2WRGV4W80BKD6nyccaCW2qgt84734pKNpkOeUpG92Tpofdq+OVKj1hkf8VeS6u0LRLq80oUuGN2aFHdo7SVLRhIh4KBWJj092Ih3jaSCptbMBoLZNIiL8GMyFvGhPCyfDWKaJ9MBcVJEXmKTQtaTZAgNQKsUDpfML+1vKJtMCxMwnOwfAIIc8zSxvI6HzQsPaTJxxkLbuRpTXFbYSM3pl6+3br+ydJD22pP3Z35gvn3GyITzX96Fnqus/dbZKZeEFOKig9tajnulrZggs+PCYUM9+JEqRzJmTK8saFHHKJojnJpEf+lZdTYxrq/KbH7h5xtGAqFZRI6UIMFLSZIkBoBlihZO7IlxrBXNW+sLMscLJ+Zt7moPGlySPb9VpUxcIMIa+oRr+f56/CucF7dS+m9svPn+8KTJofE8jneNy3dgMmBHjFfksLtL2fEiLkxS+KNdswOiSmUotPOMRFha1iQxXwUjRfd47OmU3ySYKjUgtkcpROxrJgznVBmEmrt5W9DJBeidDionGK7lk2KLIHUCLBAaTWFle2FNW2d7Mrqhrrd+erq5vLm2uoC33PN5ND2jmwvMa6qrWlccGN1e2GLT49RXnJ1Y01s97axulZ40uTQgjDwlnqSwxbcfL+6wM22vrol34ZPF1/d4QupTQ5tLcjLk5jsTdyUFnSWdNn8RK0zZTgZaxZEagRYwKPKWqV4XFnLBI8ra5ldUOtZMJijhIUfzP77BiMWCDr944BdCSxYFliwfpLEEyXVfzBvqrsCIIAFywILOvDr1RaxkKSFBYGnwIJlaUILuk4TVdwHoFH4yoLbzWbBTVhQ/mCwIPCQRlhwtbDOoDyNsOCCrZUEDbCgpX8fGgcsCHxJpLAc2DUsdUSThmrN7rBo04Jb9b9lZd7Dgl6+LwCSEK3V/22uzIJcg1+ZLNF6/W9VGXvftqi6JtA97A1Uug4sCPxJ2LCJhivkLYVg7ut4xd63rY3y9b9nZRZgQS/fFwBJ1vUu8Za1Kfzu63jHyvhkgQzR+/rftBLrogJ98wALAn8SI5c7fpvz1gKerNvSse2cqNtetqll14EFgU9JumyfHau/41HKb9b/dmXZyFPI3ieTLd1U1FG2SduyqDmABYFfibqZDOX7AlvIjXCCRPPuJUjWF+1/13L6SrKOs9p0JU9gQeBXgiGiJXfGBjf4Ju0WJShK1uVX3QkH13mtK9tjcDH2D8SCO5MGN7fmLS2xbSiwIPAtGfZlp4XVtRWHWRO1B21M0ct+wDdQWt1y+DrebzPjULqGgpsxFg3S4uqq45+MKFMYbjIJMhu1WanU4jiwIPCeWDLk0oqsUM5WcaNM2K2lYelkbcqJp926olzzffPd+lGr03yfBfAh8XDK8e97Omw//Mq05VKOKzmaqKPQXDaRijp9QaFcvNniQI7TP6Z1YEEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgO/4/ZNS8qnrP1c0AAAAldEVYdGRhdGU6Y3JlYXRlADIwMjMtMDYtMTJUMjE6MjQ6MTYrMDA6MDBqOx+2AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDIzLTA2LTEyVDIxOjI0OjE2KzAwOjAwG2anCgAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyMy0wNi0xMlQyMToyNDoxNiswMDowMExzhtUAAAAASUVORK5CYII=)

The black connections are using the Mish activation function (which has been shown to provide a consistent improvement in performance), the blue use LeakyReLU, green uses Sigmoid and red has no activation function.

Another relevant thing to point out is the use of kaiming normal initialization for the convolutions. It showed better results compared to using the default pytorch initialization only if the convolution is followed by a LeakyReLU-type activation function


```python
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, trainable_backbone_layers=5)
model2 = fasterrcnn_resnet50_fpn_v2(weights=weights, trainable_backbone_layers=5)

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
                                nn.Mish(inplace=True),
                                nn.Linear(4096, 8192),
                                nn.Mish(inplace=True),
                                nn.Linear(8192, 8192),
                                nn.Mish(inplace=True),
                                nn.Linear(8192, 2048),
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

        self.up5 = nn.Sequential(conv(256, 256,3,1,1,padding_mode='reflect'),
                                nn.Mish(inplace=True),
                                nn.PixelShuffle(2))
        self.up4 = nn.Sequential(conv(256+64, 256,3,1,1,padding_mode='reflect'),
                                nn.Mish(inplace=True),
                                nn.PixelShuffle(2))


        self.idk = model.backbone
        self.lol = model2.backbone

    def forward(self,x, bertx):
        # Pass the image though the attention blocks
        x = self.idk(x) #Initial Decomposition Kernels
        #Extract the outpus
        x1 = x.popitem(last=False)[1]
        x2 = x.popitem(last=False)[1]
        x3 = x.popitem(last=False)[1]
        x4 = x.popitem(last=False)[1]
        x5 = x.popitem(last=False)[1]

        #Upsampling block
        x5 = self.up5(x5) #256,2,2 -> 64,4,4
        x4 = torch.cat((x4,x5), dim=1) # 256+64,4,4
        x4 = self.up4(x4) #320,4,4 -> 64,8,8
        x3 = torch.cat((x3,x4), dim=1) # 256+64,8,8
        x3 = self.up4(x3) #320,8,8 -> 64,16,16
        x2 = torch.cat((x2,x3), dim=1) # 256+64,16,16
        x2 = self.up4(x2) #320,16,16 -> 64,32,32
        x1 = torch.cat((x1,x2), dim=1) # 256+64,32,32

        #Residual encoder blocks 320 -> 9
        x = self.r1(x1)
        x = self.r2(x) + x
        x = self.r3(x)
        x = self.r4(x) + x
        x = self.r5(x)
        x = self.r6(x) + x
        x = self.r7(x)

        #Prepare image to be joined with bert
        top_left = x[:, :3, :, :]
        top_right = x[:, 3:6, :, :]
        bottom_left = x[:, 6:, :, :]

        #Process bert embeddings as a matrix
        bertx = self.L2(bertx) #768 -> 1024
        bertx = bertx.reshape(-1, 3, int(resize/4), int(resize/4))

        #Join the data
        xx = torch.zeros((bertx.shape[0], 3, int(resize/2), int(resize/2)), device=torch.device(device))#.to(device)
        xx[:, :, :int(resize/4), :int(resize/4)] = top_left
        xx[:, :, :int(resize/4), int(resize/4):] = top_right
        xx[:, :, int(resize/4):, :int(resize/4)] = bottom_left
        xx[:, :, int(resize/4):, int(resize/4):] = bertx

        #Pass the joint data into the attention blocks
        x = self.lol(xx) #Link Output Labels
        #Extract the outputs
        x1 = x.popitem(last=False)[1]
        x2 = x.popitem(last=False)[1]
        x3b = x.popitem(last=False)[1]
        x4 = x.popitem(last=False)[1]

        #Upsampling block with jump connection x2 + x3
        x4 = self.up5(x4)
        x3b = torch.cat((x3b,x4), dim=1)
        x3b = self.up4(x3b)
        x2 = torch.cat((x2,x3b), dim=1)
        x2 = self.up4(x2) + x3
        x1 = torch.cat((x1,x2), dim=1)

        #Compress the data
        x = self.idk2(x1) # 320,16,16 -> 9,16,16
        #Flatten and final fully connected layer
        x = self.flat(x)
        x = self.L3(x)
        #output is between [0,1] so we multiply it by resize-1
        return x*(resize-1)
```

    Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth" to /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
    100%|██████████| 167M/167M [00:00<00:00, 238MB/s]


## Loss

For the loss we first tried only using a iou based loss available in pytorch, which goes from 0 to 2, but this led to poor performance. We also tried using only a corner based loss where we use the MSE of the goal corners of the bounding box with the model output and it performed better that only the iou loss but still wasn't optimal.

We decided to combine both, initially using a geometric mean between the means of both losses, then we switched to using the sum of each bbox error for both losses and still use the geometric mean, and for the loss below we use a simple multiplication divided by the size of the input image and the batch size to normalize it up to a point and be able to compare it with results that use different batch sizes and image sizes.


```python
criterion = nn.MSELoss()

def final_loss(outputs, bbox):
    #Get the MSE values of each corner of the bbox
    loss = 0
    for i in range(0,4):
        loss += criterion(outputs.transpose(1,0)[i], bbox.transpose(1,0)[i])

    #Get the iou loss and ignore the iou's of boxes that are negative and use the mean instead
    neg = tvo.generalized_box_iou_loss(outputs,bbox)
    if (neg < 0).any():
        mask = neg >= 0
        pos = torch.masked_select(neg, mask)
        l1 = torch.mean(pos)
    else:
        l1 = tvo.generalized_box_iou_loss(outputs,bbox,reduction='sum')
    l2 = loss
    #Make the loss multi-objective and normalize it
    return (l1*l2)/(resize*bbox.size(0))
```

## Training & Testing


For training we used AdamW as it has shown consistent better performance over Adam. We also use gradient clipping to combat exploding gradients.

Regarding data augmentation, we tried implementing color jitter which is known for increasing performance but in this case the model was severly affected by it, and any other type of value shifting augmentation for that matter.

We decided to use a batch size 32 because it was the limit our GPU allowed us to use. The size of the images was set to 256 because it presented a clear advantage over 128 (it seems the bigger the image the better performance gets). For the model to run without needing specific changes in the sizes of the tensors in the structure the model needs to receive as input images with sizes of base 2.

IT MAY CRASH FOR NO REASON DUE TO COLAB -_-


```python
if True:
    torch.backends.cudnn.benchmark = True

    train_ds = loadData(path, 'train/')
    val_ds = loadData(path, 'val/')
    test_ds = loadData(path, 'test/')

    print(len(train_ds), len(val_ds), len(test_ds))

    train_dl, train_length = create_data_loader_CustomDataset(train_ds, batch_size, eval=False)
    val_dl, train_length = create_data_loader_CustomDataset(val_ds, batch_size, eval=True)
    test_dl, train_length = create_data_loader_CustomDataset(test_ds, 1, eval=True)

    #Model
    boxfinder = BXfinder();
    boxfinder.to(device)
    #Load the pretrained backbones
    model.backbone.to(device)
    model2.backbone.to(device)
    optimizer = torch.optim.AdamW(boxfinder.parameters(), lr=init_lr)

    #Training and Validation loops
    elapsed_time=0
    for epoch in range(1, n_epochs+1):
        start_time = time.time()
        train_loss = 0.0
        val_loss= 0.0
        print(epoch, 'ETA: ' + str(datetime.timedelta(seconds = elapsed_time*n_epochs+1-epoch)))

        #Training loop
        boxfinder.train()
        i=0
        iou = []
        for bi, data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):

            images,label,bbox = data['img'], data['label'][0], data['bbox']
            images = images.to(device)
            bbox = bbox.to(device)
            #Turn the bbox data from 2 coordinates + width and heigth to 4 coordinates
            bbox[:,2] = bbox[:,0]+bbox[:,2]
            bbox[:,3] = bbox[:,1]+bbox[:,3]

            recImgs = images
            bertLabels = data['blabel']
            images.requires_grad=True
            bertLabels.requires_grad=True
            bbox.requires_grad=True

            #Process the data
            outputs = boxfinder(recImgs, bertLabels.to(device))

            #Loss and backprop
            loss = final_loss(outputs, bbox)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(boxfinder.parameters(), clipping_value)
            optimizer.step()
            train_loss += loss.item()*images.size(0)

            #Calculate iou
            for j in range(0,images.size(0)):
                temp = tvo.box_iou(bbox[j].unsqueeze(0),outputs[j].unsqueeze(0)).to('cpu')
                iou.append(temp.item())


            i+=1
            #break

        train_loss = train_loss/len(train_dl)
        print('E: {} T Loss: {:.3f}'.format(epoch, train_loss) + " %" + "{:.3}".format(np.exp(-abs(train_loss))*100))
        print(np.array(iou).mean())
        #Save model weights
        if epoch%save_freq == 0:
            try:
                torch.save(boxfinder.state_dict(), spath + 'epoch{0:05d}.pth'.format(epoch))
            except Exception as e:
                print("An error occurred:", e)

        #Validation loop
        boxfinder.eval()
        iou=[]
        with torch.no_grad():
            for bi, data in tqdm(enumerate(val_dl), total=int(len(val_ds)/val_dl.batch_size)):
                images,label,bbox = data['img'], data['label'][0], data['bbox']
                images = images.to(device)
                bbox = bbox.to(device)
                #Turn the bbox data from 2 coordinates + width and heigth to 4 coordinates
                bbox[:,2] = bbox[:,0]+bbox[:,2]
                bbox[:,3] = bbox[:,1]+bbox[:,3]

                recImgs = images
                bertLabels = data['blabel']

                #Process the data
                outputs = boxfinder(recImgs, bertLabels.to(device))

                #Calculate loss (used for plotting)
                loss = loss = final_loss(outputs, bbox)
                val_loss += loss.item()*images.size(0)

                #iou
                for j in range(0,images.size(0)):
                    temp = tvo.box_iou(bbox[j].unsqueeze(0),outputs[j].unsqueeze(0)).to('cpu')
                    iou.append(temp.item())


            val_loss = val_loss/len(val_dl)
            print('E: {} V Loss: {:.3f}'.format(epoch, val_loss) + " %" + "{:.3}".format(np.exp(-abs(val_loss))*100))
            print(np.array(iou).mean())
            print(outputs[0], bbox[0])


        elapsed_time = time.time() - start_time

    #Test loop
    with torch.no_grad():

        #Calculate test IOU
        iou = []
        for data in test_dl:
            images,label,bbox = data['img'], data['label'][0], data['bbox']
            bertLabels = data['blabel']
            images = images.to(device)
            bbox = bbox.to(device)
            bbox[:,2] = bbox[:,0]+bbox[:,2]
            bbox[:,3] = bbox[:,1]+bbox[:,3]

            outputs = boxfinder(images, bertLabels.to(device))
            temp = tvo.box_iou(bbox,outputs).to('cpu')
            iou.append(temp.item())
        print(np.array(iou).mean())

```

    1000 515 2600
    1 ETA: 0:00:00


    32it [03:43,  6.98s/it]


    E: 1 T Loss: 1816.266 %0.0
    0.19842453765689788


    17it [10:21, 36.53s/it]


    E: 1 V Loss: 1546.095 %0.0
    0.21354853074355684
    tensor([ 57.5876,  45.6722, 180.7797, 207.4397], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    2 ETA: 11:43:53.943104


    32it [01:17,  2.44s/it]


    E: 2 T Loss: 1311.808 %0.0
    0.2549892421023542


    17it [00:25,  1.48s/it]


    E: 2 V Loss: 1231.470 %0.0
    0.2546104815694757
    tensor([ 53.3259,  58.2527, 186.8239, 226.9981], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    3 ETA: 1:25:51.043890


    32it [01:20,  2.51s/it]


    E: 3 T Loss: 1031.768 %0.0
    0.2941617276710021


    17it [00:25,  1.53s/it]


    E: 3 V Loss: 1090.956 %0.0
    0.2712725356492458
    tensor([ 49.0958,  45.7359, 176.6604, 223.5040], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    4 ETA: 1:28:41.288261


    32it [01:16,  2.39s/it]


    E: 4 T Loss: 699.298 %1.99e-302
    0.3447650621999055


    17it [00:25,  1.47s/it]


    E: 4 V Loss: 1279.488 %0.0
    0.2674680294532293
    tensor([ 74.6308,  38.0828, 200.7401, 238.8118], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    5 ETA: 1:24:29.752904


    32it [01:14,  2.33s/it]


    E: 5 T Loss: 532.401 %6.04e-230
    0.3744297720845789


    17it [00:27,  1.62s/it]


    E: 5 V Loss: 1163.094 %0.0
    0.25810114163498493
    tensor([ 18.1771,  35.9677, 121.2862, 243.2894], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    6 ETA: 1:50:20.689530


    32it [01:16,  2.38s/it]


    E: 6 T Loss: 390.747 %2e-168
    0.41600797688215974


    17it [00:24,  1.44s/it]


    E: 6 V Loss: 1126.199 %0.0
    0.2672340792029626
    tensor([ 35.0002,  24.4218, 180.3736, 247.1371], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    7 ETA: 1:23:48.464514


    32it [01:16,  2.40s/it]


    E: 7 T Loss: 257.485 %1.5e-110
    0.46070596234407274


    17it [00:24,  1.46s/it]


    E: 7 V Loss: 1020.615 %0.0
    0.29435125455300964
    tensor([ 22.9799,  19.1150, 203.0066, 252.5690], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    8 ETA: 1:24:33.974829


    32it [01:15,  2.34s/it]


    E: 8 T Loss: 198.771 %4.73e-85
    0.49466306571336466


    17it [00:24,  1.43s/it]


    E: 8 V Loss: 1076.603 %0.0
    0.2868264392587962
    tensor([ 15.7289,  30.4166, 209.4583, 248.0829], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    9 ETA: 1:22:42.657384


    32it [01:16,  2.39s/it]


    E: 9 T Loss: 157.897 %2.67e-67
    0.5164001286588609


    17it [00:24,  1.46s/it]


    E: 9 V Loss: 980.817 %0.0
    0.2957998336916699
    tensor([ 29.4506,  20.1868, 173.8939, 251.9702], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    10 ETA: 1:24:13.114990


    32it [01:15,  2.36s/it]


    E: 10 T Loss: 139.011 %4.25e-59
    0.5299703669715673


    17it [00:29,  1.76s/it]


    E: 10 V Loss: 987.390 %0.0
    0.28247375074475645
    tensor([ 32.2810,  30.0143, 225.8207, 245.5482], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    11 ETA: 1:55:43.435624


    32it [01:15,  2.35s/it]


    E: 11 T Loss: 119.728 %1.01e-50
    0.5510229022097773


    17it [00:24,  1.45s/it]


    E: 11 V Loss: 997.155 %0.0
    0.2995774067087076
    tensor([ 47.6840,  59.9443, 201.5637, 250.3138], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    12 ETA: 1:23:05.838415


    32it [01:13,  2.30s/it]


    E: 12 T Loss: 87.610 %8.95e-37
    0.5769954618141055


    17it [00:24,  1.43s/it]


    E: 12 V Loss: 1003.262 %0.0
    0.2851160091415807
    tensor([ 33.4414,  59.4021, 186.9269, 246.7729], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    13 ETA: 1:21:22.562137


    32it [01:13,  2.30s/it]


    E: 13 T Loss: 75.929 %1.06e-31
    0.5917040042579174


    17it [00:26,  1.54s/it]


    E: 13 V Loss: 1080.538 %0.0
    0.2686163086460528
    tensor([ 32.6670,  53.2916, 208.3098, 252.2400], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    14 ETA: 1:22:52.681641


    32it [01:15,  2.37s/it]


    E: 14 T Loss: 63.857 %1.85e-26
    0.6058609468579526


    17it [00:25,  1.49s/it]


    E: 14 V Loss: 946.481 %0.0
    0.30640262235679383
    tensor([ 32.6266,  63.1182, 220.0849, 253.7064], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    15 ETA: 1:24:07.330557


    32it [01:16,  2.38s/it]


    E: 15 T Loss: 56.268 %3.66e-23
    0.6225280732810498


    17it [00:30,  1.77s/it]


    E: 15 V Loss: 927.090 %0.0
    0.30503585939770295
    tensor([ 53.0096,  74.3189, 204.0885, 253.4353], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    16 ETA: 1:32:29.969802


    32it [01:16,  2.40s/it]


    E: 16 T Loss: 46.813 %4.67e-19
    0.6380156720429659


    17it [00:25,  1.49s/it]


    E: 16 V Loss: 940.978 %0.0
    0.2933993055788095
    tensor([ 15.6752,  71.9093, 212.6670, 252.0148], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    17 ETA: 1:24:48.718435


    32it [01:16,  2.39s/it]


    E: 17 T Loss: 43.046 %2.02e-17
    0.6450108747500927


    17it [00:25,  1.48s/it]


    E: 17 V Loss: 962.856 %0.0
    0.2886012577759192
    tensor([ 63.8662,  63.0528, 214.8842, 253.6823], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    18 ETA: 1:24:33.350866


    32it [01:15,  2.35s/it]


    E: 18 T Loss: 45.892 %1.17e-18
    0.6457367052361369


    17it [00:25,  1.50s/it]


    E: 18 V Loss: 942.268 %0.0
    0.28902810185614214
    tensor([ 51.4621,  44.3992, 233.1357, 252.4743], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    19 ETA: 1:23:40.824761


    32it [01:15,  2.35s/it]


    E: 19 T Loss: 32.985 %4.73e-13
    0.6730546181816608


    17it [00:24,  1.46s/it]


    E: 19 V Loss: 889.677 %0.0
    0.3097441684929383
    tensor([ 40.1527,  48.3122, 223.6103, 252.8563], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    20 ETA: 1:23:08.290924


    32it [01:14,  2.33s/it]


    E: 20 T Loss: 31.343 %2.44e-12
    0.6748368531614543


    17it [00:27,  1.62s/it]


    E: 20 V Loss: 947.887 %0.0
    0.2887761476822932
    tensor([ 43.2073,  63.8864, 222.8810, 251.0521], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    21 ETA: 1:31:45.609767


    32it [01:16,  2.38s/it]


    E: 21 T Loss: 24.519 %2.25e-09
    0.6961862015109509


    17it [00:25,  1.51s/it]


    E: 21 V Loss: 1012.771 %0.0
    0.28664261702068367
    tensor([ 54.0640,  55.9218, 211.5570, 252.4439], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    22 ETA: 1:24:28.180386


    32it [01:15,  2.35s/it]


    E: 22 T Loss: 28.002 %6.9e-11
    0.6857493699043989


    17it [00:25,  1.47s/it]


    E: 22 V Loss: 880.909 %0.0
    0.32330192073920255
    tensor([ 53.2075,  54.3316, 205.0022, 250.0536], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    23 ETA: 1:23:17.698656


    32it [01:14,  2.34s/it]


    E: 23 T Loss: 23.501 %6.22e-09
    0.6980511584430933


    17it [00:24,  1.45s/it]


    E: 23 V Loss: 907.092 %0.0
    0.3016713041779404
    tensor([ 59.4586,  57.3351, 212.8973, 253.7533], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    24 ETA: 1:22:42.939491


    32it [01:14,  2.34s/it]


    E: 24 T Loss: 21.790 %3.44e-08
    0.7045550287514925


    17it [00:24,  1.45s/it]


    E: 24 V Loss: 896.365 %0.0
    0.3054440285269063
    tensor([ 35.8803,  70.2016, 184.9290, 252.3590], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    25 ETA: 1:22:27.457529


    32it [01:13,  2.31s/it]


    E: 25 T Loss: 20.074 %1.91e-07
    0.7089782227128744


    17it [00:28,  1.70s/it]


    E: 25 V Loss: 922.078 %0.0
    0.29138064439023065
    tensor([ 54.6428,  51.4784, 196.7452, 252.2608], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    26 ETA: 1:36:19.585886


    32it [01:16,  2.38s/it]


    E: 26 T Loss: 16.991 %4.18e-06
    0.7255696358829736


    17it [00:24,  1.44s/it]


    E: 26 V Loss: 925.713 %0.0
    0.301008249222961
    tensor([ 50.3693,  84.9014, 211.6017, 252.3354], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    27 ETA: 1:23:27.376932


    32it [01:13,  2.30s/it]


    E: 27 T Loss: 17.185 %3.44e-06
    0.7236090155988931


    17it [00:25,  1.48s/it]


    E: 27 V Loss: 909.327 %0.0
    0.30765314704664526
    tensor([ 60.5892,  65.1556, 192.9084, 251.5121], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    28 ETA: 1:21:47.777451


    32it [01:15,  2.36s/it]


    E: 28 T Loss: 14.889 %3.42e-05
    0.7357279886975885


    17it [00:24,  1.46s/it]


    E: 28 V Loss: 895.829 %0.0
    0.3081682808579678
    tensor([ 36.1237,  58.2851, 195.9612, 249.3157], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    29 ETA: 1:23:01.963083


    32it [01:15,  2.37s/it]


    E: 29 T Loss: 12.050 %0.000584
    0.7509382752478123


    17it [00:24,  1.45s/it]


    E: 29 V Loss: 930.704 %0.0
    0.3075125436181004
    tensor([ 32.3438,  72.4545, 207.4681, 251.4287], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    30 ETA: 1:23:21.730450


    32it [01:14,  2.34s/it]


    E: 30 T Loss: 15.224 %2.44e-05
    0.7368741952329874


    17it [00:28,  1.69s/it]


    E: 30 V Loss: 859.046 %0.0
    0.33123707968416144
    tensor([ 49.4310,  60.8260, 219.0392, 253.4873], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    31 ETA: 1:32:34.405661


    32it [01:14,  2.32s/it]


    E: 31 T Loss: 11.356 %0.00117
    0.753354505032301


    17it [00:24,  1.46s/it]


    E: 31 V Loss: 898.385 %0.0
    0.30943795752392866
    tensor([ 42.2699,  58.4175, 199.6068, 253.3885], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    32 ETA: 1:22:12.024509


    32it [01:13,  2.31s/it]


    E: 32 T Loss: 8.686 %0.0169
    0.7743297455310821


    17it [00:25,  1.48s/it]


    E: 32 V Loss: 867.894 %0.0
    0.3062105230797191
    tensor([ 64.7488,  72.7609, 180.2898, 250.6459], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    33 ETA: 1:22:06.591008


    32it [01:15,  2.37s/it]


    E: 33 T Loss: 9.794 %0.00558
    0.7671236404329538


    17it [00:23,  1.40s/it]


    E: 33 V Loss: 908.905 %0.0
    0.3129988767179543
    tensor([ 49.6085,  67.4261, 225.5719, 250.2109], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    34 ETA: 1:22:29.340753


    32it [01:14,  2.33s/it]


    E: 34 T Loss: 8.080 %0.031
    0.7760590347293764


    17it [00:26,  1.54s/it]


    E: 34 V Loss: 839.301 %0.0
    0.3224380679812861
    tensor([ 51.6456,  51.9194, 200.7109, 252.2681], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    35 ETA: 1:23:23.229240


    32it [01:15,  2.35s/it]


    E: 35 T Loss: 7.829 %0.0398
    0.7799926450103521


    17it [00:28,  1.66s/it]


    E: 35 V Loss: 863.853 %0.0
    0.31303563844695126
    tensor([ 39.8071,  60.8070, 184.3189, 253.8100], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    36 ETA: 1:30:37.817838


    32it [01:14,  2.33s/it]


    E: 36 T Loss: 6.412 %0.164
    0.789387505069375


    17it [00:24,  1.45s/it]


    E: 36 V Loss: 873.273 %0.0
    0.31365956274195783
    tensor([ 58.7613,  64.0495, 204.6152, 250.2114], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    37 ETA: 1:22:05.285043


    32it [01:15,  2.35s/it]


    E: 37 T Loss: 6.063 %0.233
    0.7953827947676182


    17it [00:24,  1.46s/it]


    E: 37 V Loss: 884.843 %0.0
    0.31115879038443733
    tensor([ 43.4182,  61.5672, 224.3466, 251.8367], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    38 ETA: 1:22:53.935009


    32it [01:16,  2.40s/it]


    E: 38 T Loss: 5.801 %0.302
    0.7960626593828202


    17it [00:25,  1.51s/it]


    E: 38 V Loss: 882.068 %0.0
    0.32386362726429424
    tensor([ 49.0703,  49.9348, 227.6774, 251.2114], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    39 ETA: 1:24:46.793792


    32it [01:16,  2.39s/it]


    E: 39 T Loss: 5.434 %0.437
    0.8015984204113483


    17it [00:25,  1.49s/it]


    E: 39 V Loss: 851.828 %0.0
    0.31623443313863003
    tensor([ 54.2905,  60.0518, 208.3240, 252.0570], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    40 ETA: 1:24:12.873562


    32it [01:16,  2.39s/it]


    E: 40 T Loss: 5.700 %0.335
    0.7975444615483284


    17it [00:28,  1.70s/it]


    E: 40 V Loss: 865.568 %0.0
    0.3154764073426175
    tensor([ 42.5968,  64.5233, 214.9990, 253.7706], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    41 ETA: 1:35:55.956278


    32it [01:18,  2.46s/it]


    E: 41 T Loss: 4.790 %0.831
    0.807704634770751


    17it [00:24,  1.46s/it]


    E: 41 V Loss: 850.494 %0.0
    0.3289916279192254
    tensor([ 55.4637,  62.0384, 230.7046, 253.5389], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    42 ETA: 1:25:38.615450


    32it [01:15,  2.35s/it]


    E: 42 T Loss: 4.110 %1.64
    0.8142933030724525


    17it [00:25,  1.47s/it]


    E: 42 V Loss: 856.039 %0.0
    0.3114539720818233
    tensor([ 52.8460,  58.1133, 210.6460, 253.3961], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    43 ETA: 1:22:59.936095


    32it [01:16,  2.39s/it]


    E: 43 T Loss: 4.383 %1.25
    0.8138212809562683


    17it [00:25,  1.53s/it]


    E: 43 V Loss: 871.983 %0.0
    0.30460038974720677
    tensor([ 52.4111,  58.4344, 203.9833, 252.1824], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    44 ETA: 1:24:46.657865


    32it [01:16,  2.40s/it]


    E: 44 T Loss: 3.848 %2.13
    0.8211914355754852


    17it [00:26,  1.55s/it]


    E: 44 V Loss: 859.623 %0.0
    0.3129385929395654
    tensor([ 50.2868,  57.9362, 202.8764, 252.2140], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    45 ETA: 1:25:21.512049


    32it [01:16,  2.38s/it]


    E: 45 T Loss: 3.913 %2.0
    0.8198689674735069


    17it [00:59,  3.48s/it]


    E: 45 V Loss: 834.485 %0.0
    0.3223262185013844
    tensor([ 42.1886,  55.6666, 211.4087, 253.2565], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    46 ETA: 1:57:50.546834


    32it [01:15,  2.36s/it]


    E: 46 T Loss: 3.807 %2.22
    0.8198583938777447


    17it [00:26,  1.53s/it]


    E: 46 V Loss: 820.596 %0.0
    0.3319921846334253
    tensor([ 48.3725,  50.0261, 212.4047, 252.6401], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    47 ETA: 1:23:54.102956


    32it [01:16,  2.39s/it]


    E: 47 T Loss: 2.664 %6.97
    0.8398538560867309


    17it [00:25,  1.49s/it]


    E: 47 V Loss: 837.250 %0.0
    0.32255352956143396
    tensor([ 45.1302,  63.6346, 211.3988, 252.8839], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    48 ETA: 1:24:12.894595


    32it [01:15,  2.35s/it]


    E: 48 T Loss: 3.363 %3.46
    0.8267912602424622


    17it [00:25,  1.50s/it]


    E: 48 V Loss: 854.518 %0.0
    0.32434712325158366
    tensor([ 54.3135,  51.0646, 218.3238, 252.4913], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    49 ETA: 1:23:12.384471


    32it [01:16,  2.38s/it]


    E: 49 T Loss: 2.835 %5.87
    0.8351496394574642


    17it [00:25,  1.48s/it]


    E: 49 V Loss: 862.570 %0.0
    0.3285231804412601
    tensor([ 39.7284,  57.4514, 225.5676, 252.7890], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')
    50 ETA: 1:23:31.052991


    32it [01:15,  2.37s/it]


    E: 50 T Loss: 2.653 %7.04
    0.8384515156149864


    17it [00:30,  1.81s/it]


    E: 50 V Loss: 844.737 %0.0
    0.3237443755982948
    tensor([ 50.7501,  67.5387, 214.8286, 253.1982], device='cuda:0') tensor([ 54.6655,  64.5760, 183.2859, 253.8440], device='cuda:0')



    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-8-8bc295181004> in <cell line: 1>()
        118         #Calculate test IOU
        119         iou = []
    --> 120         for data in test_dl:
        121             images,label,bbox = data['img'], data['label'][0], data['bbox']
        122             bertLabels = data['blabel']


    /usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py in __next__(self)
        631                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
        632                 self._reset()  # type: ignore[call-arg]
    --> 633             data = self._next_data()
        634             self._num_yielded += 1
        635             if self._dataset_kind == _DatasetKind.Iterable and \


    /usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
        675     def _next_data(self):
        676         index = self._next_index()  # may raise StopIteration
    --> 677         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        678         if self._pin_memory:
        679             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)


    /usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
         49                 data = self.dataset.__getitems__(possibly_batched_index)
         50             else:
    ---> 51                 data = [self.dataset[idx] for idx in possibly_batched_index]
         52         else:
         53             data = self.dataset[possibly_batched_index]


    /usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
         49                 data = self.dataset.__getitems__(possibly_batched_index)
         50             else:
    ---> 51                 data = [self.dataset[idx] for idx in possibly_batched_index]
         52         else:
         53             data = self.dataset[possibly_batched_index]


    <ipython-input-5-3f038daa5951> in __getitem__(self, idx)
         18     def __getitem__(self, idx):
         19         with open(self.data[idx], 'rb') as file:
    ---> 20             data = pickle.load(file)
         21             if True:
         22                 label = data['label'] #take the first label raw


    OSError: [Errno 5] Input/output error


## Use the model

Remember to run the initial configurations, the dataloader and the model!


```python
loaddir =  submission_folder_path + 'epoch00100F.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
verbose = False


boxfinder = BXfinder()
boxfinder = boxfinder.to(device)
boxfinder.load_state_dict(torch.load(loaddir, map_location=torch.device(device)))
boxfinder.eval()

test_ds = loadData(path, 'test/')
test_dl, train_length = create_data_loader_CustomDataset(test_ds, 1, eval=False)
#NEED TO ALSO LOAD OG IMAGES FOR PLOTTING

mean = [0.5, 0.5, 0.5]  # Mean values for each channel
std = [0.5, 0.5, 0.5]   # Standard deviation values for each channel

denormalize = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=1 / np.array(std)),
    transforms.Normalize(mean=-np.array(mean), std=[1, 1, 1])
])


with torch.no_grad():
    iou = []
    i=0
    for data in test_dl:
        images,label,bbox = data['img'], data['label'][0], data['bbox']
        bertLabels = data['blabel']
        images = images.to(device)
        bbox = bbox.to(device)
        bbox[:,2] = bbox[:,0]+bbox[:,2]
        bbox[:,3] = bbox[:,1]+bbox[:,3]

        outputs = boxfinder(images, bertLabels.to(device))
        temp = tvo.box_iou(bbox,outputs).to('cpu')
        iou.append(temp.item())

        #Plot the image
        images = denormalize(images)
        image = images.squeeze(0).permute(2,1,0).to('cpu')
        fig, ax = plt.subplots()
        ax.imshow(image)
        plt.title(label[0])
        #Plot dataset bounding box
        x_down = data['bbox'][0][0].item()
        y_down = data['bbox'][0][1].item()
        w = data['bbox'][0][2].item()
        h = data['bbox'][0][3].item()
        if(verbose):
          print(x_down,y_down,w,h)
        rect = plt.Rectangle((x_down, y_down), w, h, linewidth=1, edgecolor='r', facecolor='none',alpha=0.3,color='r')
        ax.add_patch(rect)

        #plot model bbox
        if(verbose):
          print(outputs)
        outputs= outputs.to('cpu')
        w = outputs[:,2] - outputs[:,0]
        h = outputs[:,3] - outputs[:,1]
        if(verbose):
          print(outputs[:,0], outputs[:,1], w,h)
        rect = plt.Rectangle((outputs[:,0].item(), outputs[:,1].item()), w.item(), h.item(), linewidth=1, edgecolor='b', facecolor='none', alpha=0.3, color='b')
        ax.add_patch(rect)

        if i ==10:
          break

        i+=1
print('\n\n\nRED: Ground Truth\nBLUE: Predicted')
if(verbose):
  print(np.array(iou).mean())
```


    Output hidden; open in https://colab.research.google.com to view.


---
# Results

Overall from what you can see in the cell above, the predictions are consistent and almost every time correct. The bigger problem here is that when there are multiple instances of the same object (e.g. two zebras) that correspond to the text label it ofter select the wrong one. This is what we think it is driving down the validation loss.

Let's analyze the results more in detail.

## Results
 We tried many different architectures, parameters and models to see which was the one performing better.
 The best runs can be visualized on Weight AND Biases. We deleted all the results that were showing that a particular model was not converging at all to keep it clean. In the following table we can see:


1.   **final norm 256 x2jump2** : This is the final model which all the configurations that can be found in the code above
2.   **mod4v2Sig-256** : This model doesn't have the x3 jump connection nor normalization of the input data
3.   **RCNN-Clip_Bert** : This structure uses only the initial attention blocks, it reduced the processed image matrix and flattened it to concatenate it with the bert embeddings and go directly to a fully connected layer. This model and the ones below were trained on 128 size images and have the same architecture
4.   **RCNN-Clip_NOBert** : Removed Bert from the model
5.   **RCNN_NO_Clip** : Removed the clip preprocessing from the images


| MODEL NAME                | Train IOU | Val IOU | Train Loss | Val Loss |
| :------------------------ | :-------: | ------: | ---------: | -------: |
| final norm 256 x2jump     |   0,9369  | 0,4455  | 0,2622     | 804,616  |
| mod4v2Sig-256             |   0,9354  | 0,4251  | 0,3303     | 893,239  |
| RCNN-Clip-Bert            |   0,9086  | 0,3301  | 7,707      | 331.181  |
| RCNN-Clip-NOBert          |   -       | -       | 265,239    | 893,239  |
| RCNN-NO-Clip              |   -       | -       | 8,527      | 419,313  |


Note: The loss used between models was changed and it's affected by the size of the input image

Even if the loss is not perfectly comparable, from those value we get a strong message: the CLIP preprocessing information is helping the model achieve better results **but** **BERT** is helping way more.



## Links
Our WANDB page can be found here:
https://wandb.ai/unitnais/visual-grounding.

Our GitHub repository can be found here:
https://github.com/debryu/visual-grounding.


---


# Future Work
Our project is far from perfect and actually we have some ideas on how we would like to fix some problems we encountered.
First, from the results we can see that the model is doing very good on the train set but it cannot generalize well and does poorly on the validations.

## Idea
One idea to have a more robust model is to approach the problem in a similar fashion as Language Models do in Masked Language Modeling:
the model produce the logits (the predicted probability for a specific token) and the loss is computed by the CROSS ENTROPY between the logits and the one hot encoding of the ground truth.
In the same way we could chunk the image into blocks as we did for computing the CLIP similarity score matrix and use this to predict a matrix of "logits" where each one describes the probability of that particular block to contain the object.

The loss is computed by the CROSS ENTROPY LOSS of those logits between the ground truth matrix which is a matrix composed by 0 if the block does not contain the bounding box of the object, 1 if it does, and 0.5 if the bounding box cuts (it does not full cover) the block.

Basically, this model will learn to predict a mask for the image that finds which element in the grid is relevant to the queried text. This is not really the same as we were doing with the CLIP similarity score because as we saw it cannot handle "negative prompts" (that are not present in the image) and also it is not a true mask layer (but just makes the image pixels go to 0 if they are not relevant).

## Things we might have done wrong

A posteriori if we were able to go back in time we would have changed little things that now we deem are actually holding us back.
One is the way we added the CLIP similarity score in the model, since we now believe that a better way to do it is to have the raw similarity matrix in the dataset and combine it in different ways at different steps instead of just applying it directly to the image.
In this way the model could have "jump connections" and access the raw CLIP similarity matrix at deeper layers for example.

The second important thing we noted could be problematic is the loss function: it was really hard to design one that would have not caused problems because of the generalized IOU returning values that pushed the model to predict bounding boxes as big as they could be.
We fixed that by weighting it by another loss that takes into consideration the distance between the corners. But of course, this could be further improved and it is not really a good solution.

## Other ideas

Another way of solving some issues, in particular the overfitting, is to either/both:
1.  Augment the dataset (color jittering, crop, rotation, etc.)
2.  Increase the model size and number of parameters

Another possibility is:
3.  Maybe CLIP is the bottleneck that is preventing the model to generalize

Actually we tried augmenting the dataset but we got worse results (so we did not spent too much time on it).
On the other hand, increasing the model size was not feasible since we have 12GB of VRAM and so we cannot run bigger models.

# Citations

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

---

[Benchmarking Detection Transfer Learning with Vision Transformers](https://arxiv.org/pdf/2111.11429.pdf)

---

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

---

[Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/pdf/1902.09630.pdf)

---

[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158.pdf)

More on ESPCN:
[An Overview of ESPCN: An Efficient Sub-pixel Convolutional Neural Network](https://medium.com/@zhuocen93/an-overview-of-espcn-an-efficient-sub-pixel-convolutional-neural-network-b76d0a6c875e)

---

[Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101.pdf)

---

[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158.pdf)

---

[Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/pdf/1908.08681v1.pdf)

More on MISH:
[Meet Mish — New State of the Art AI Activation Function. The successor to ReLU?](https://lessw.medium.com/meet-mish-new-state-of-the-art-ai-activation-function-the-successor-to-relu-846a6d93471f)

---

[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852v1.pdf)

More on Kaiming Initialization:
[Weight initialization for CNNs: A Deep Dive into He Initialization](https://medium.com/@tylernisonoff/weight-initialization-for-cnns-a-deep-dive-into-he-initialization-50b03f37f53d)

