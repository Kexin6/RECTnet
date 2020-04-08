# %%
# REVIEW import packages
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # for plotting
import torch.optim as optim #for gradient descent
from torchvision import datasets, transforms
import torchvision.models

torch.manual_seed(1) # set the random seed

# %%
# REVIEW load model
classes = ['0: Neutral', '1: Happiness', 
           '2: Sadness', '3: Surprise', 
           '4: Fear', '5: Disgust', '6: Anger']
tensor_path = "final_model_tensor/model_FaceRec_bs128_lr0.001_epoch6"
model = torchvision.models.googlenet(pretrained=True)
model.fc = nn.Linear(1024, 7)
torch.set_flush_denormal(True)
model.load_state_dict(torch.load(tensor_path, map_location=torch.device('cpu')))
model.eval()
def run_model(input):
    with torch.no_grad():
        out = model(input)
        return out

# %%
# REVIEW load sample data
from pprint import pprint

data_path = "/Users/rain/Documents/APS360/project/dataset_valid"
data_transform = transforms.Compose([transforms.Resize(224, 224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = datasets.ImageFolder(data_path, transform=data_transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                        num_workers=0, shuffle=True)

k = 0
for images, labels in loader:
    image = images[0]
    # place the colour channel at the end, instead of at the beginning
    img = np.transpose(image, [1,2,0])
    # normalize pixel intensity values to [0, 1]
    img = img / 2 + 0.5
    plt.subplot(2, 4, k+1)
    plt.axis('off')
    plt.imshow(img)
    plt.title(classes[labels[0]])
    
    output = run_model(images)
    result = torch.nn.functional.softmax(output, dim = 1)
    pred = output.max(1, keepdim=True)[1]
    # print(result, end='; ')
    print([classes[p] for p in pred], "correct" if pred[0] == labels[0] else "wrong")

    k += 1
    if k >= 8:
        break

# %%
