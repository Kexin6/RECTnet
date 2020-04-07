# -*- coding: utf-8 -*-
"""Evan - binary_happy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SDODjPZH0sJngcwMK1ostThLPCX9jLzN

## Check GPU states
"""

# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
 process = psutil.Process(os.getpid())
 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
 print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()

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

torch.manual_seed(1) # set the random seed

# obtain data
from torchvision import datasets, transforms

"""## Download datasets"""

PATH_TRAIN_IM = "/content/train_im/"
PATH_TRAIN_LD = "/content/train_ld/dataset_landmasks_auto_5600/"
PATH_VALID_IM = "/content/valid_im/dataset_valid/"
PATH_VALID_LD = "/content/valid_ld/dataset_landmasks_valid/"
PATH_CSV = "/content/csv/csv/"

PATH_TRAIN = "/content/t_im/dataset_train/"
PATH_VALID = "/content/valid_im/dataset_valid/"

def download_and_unzip(filename, target_folder="dataset") -> str:
    if filename[-4:] != ".zip": filename += ".zip"
    url = "http://yucanwu.com/aps360/" + filename

    !cd /content/
    !rm -rf unzip_temp
    !mkdir -p unzip_temp
    !wget $url
    !unzip -qq $filename -d unzip_temp/
    !rm -rf $target_folder
    !mkdir -p $target_folder
    !mv unzip_temp/* $target_folder

    return "/content/" + target_folder + "/"

!rm -rf /content/*
#PATH_TRAIN_IM = download_and_unzip("set_a_5600", "train_im")
#PATH_TRAIN_LD = download_and_unzip("set_al_5600", "train_ld") + "dataset_landmasks_auto_5600/"

PATH_VALID_IM = download_and_unzip("set_v_3500", "valid_im") + "dataset_valid/"
#PATH_VALID_LD = download_and_unzip("set_vl_3500", "valid_ld") + "dataset_landmasks_valid/"

# PATH_SMALL_IM = download_and_unzip("set_a_35", "small_im")
# PATH_SMALL_LD = download_and_unzip("set_al_35", "small_ld") + "dataset_landmasks_auto_35/"

#PATH_T_IM = download_and_unzip("set_t_283901", "t_im")
PATH_EXTRA_IM = download_and_unzip("set_t_358514", "t_im")
#PATH_T_LD = download_and_unzip("set_tl_283901", "t_ld") + "dataset_landmasks_train/"
#PATH_EXTRA_LD = download_and_unzip("set_tl_358514", "t_ld")

#PATH_A_IM = download_and_unzip("set_a_460382", "a_im")
#PATH_A_LD = download_and_unzip("set_al_460382", "a_ld") + "dataset_landmasks_auto/"

#PATH_CSV = download_and_unzip("csv.zip", "csv") + "csv/"

!rm *.zip

def download_and_unzip(filename, target_folder="dataset") -> str:
    if filename[-4:] != ".zip": filename += ".zip"
    url = "http://yucanwu.com/aps360/" + filename

    !cd /content/
    !rm -rf unzip_temp
    !mkdir -p unzip_temp
    !wget $url
    !unzip -qq $filename -d unzip_temp/
    !rm -rf $target_folder
    !mkdir -p $target_folder
    !mv unzip_temp/* $target_folder

    return "/content/" + target_folder + "/"
PATH_TRAIN_IM = download_and_unzip("set_a_5600", "train_im")
#PATH_TRAIN_LD = download_and_unzip("set_al_5600", "train_ld") + "dataset_landmasks_auto_5600/"

PATH_VALID_IM = download_and_unzip("set_v_3500", "valid_im") + "dataset_valid/"
#PATH_VALID_LD = download_and_unzip("set_vl_3500", "valid_ld") + "dataset_landmasks_valid/"

# PATH_SMALL_IM = download_and_unzip("set_a_35", "small_im")
# PATH_SMALL_LD = download_and_unzip("set_al_35", "small_ld") + "dataset_landmasks_auto_35/"

#PATH_T_IM = download_and_unzip("set_t_283901", "t_im")
#PATH_EXTRA_IM = download_and_unzip("set_t_358514", "t_im")
#PATH_T_LD = download_and_unzip("set_tl_283901", "t_ld") + "dataset_landmasks_train/"
#PATH_EXTRA_LD = download_and_unzip("set_tl_358514", "t_ld")

#PATH_A_IM = download_and_unzip("set_a_460382", "a_im")
#PATH_A_LD = download_and_unzip("set_al_460382", "a_ld") + "dataset_landmasks_auto/"

#PATH_CSV = download_and_unzip("csv.zip", "csv") + "csv/"

!rm *.zip

# PATH_TRAIN = "/content/train_im/"
# PATH_VALID = "/content/valid_im/dataset_valid/"

# !mv /content/train_im/2/* /content/train_im/0/
# !mv /content/train_im/3/* /content/train_im/0/
# !mv /content/train_im/4/* /content/train_im/0/
# !mv /content/train_im/5/* /content/train_im/0/
# !mv /content/train_im/6/* /content/train_im/0/

# !rm -rf /content/train_im/2
# !rm -rf /content/train_im/3
# !rm -rf /content/train_im/4
# !rm -rf /content/train_im/5
# !rm -rf /content/train_im/6

# !mv /content/valid_im/dataset_valid/2/* /content/valid_im/dataset_valid/0/
# !mv /content/valid_im/dataset_valid/3/* /content/valid_im/dataset_valid/0/
# !mv /content/valid_im/dataset_valid/4/* /content/valid_im/dataset_valid/0/
# !mv /content/valid_im/dataset_valid/5/* /content/valid_im/dataset_valid/0/
# !mv /content/valid_im/dataset_valid/6/* /content/valid_im/dataset_valid/0/

# !rm -rf /content/valid_im/dataset_valid/2
# !rm -rf /content/valid_im/dataset_valid/3
# !rm -rf /content/valid_im/dataset_valid/4
# !rm -rf /content/valid_im/dataset_valid/5
# !rm -rf /content/valid_im/dataset_valid/6

!mv /content/t_im/dataset_train/2/* /content/t_im/dataset_train/0/
!mv /content/t_im/dataset_train/3/* /content/t_im/dataset_train/0/
!mv /content/t_im/dataset_train/4/* /content/t_im/dataset_train/0/
!mv /content/t_im/dataset_train/5/* /content/t_im/dataset_train/0/
!mv /content/t_im/dataset_train/6/* /content/t_im/dataset_train/0/

!rm -rf /content/t_im/dataset_train/2
!rm -rf /content/t_im/dataset_train/3
!rm -rf /content/t_im/dataset_train/4
!rm -rf /content/t_im/dataset_train/5
!rm -rf /content/t_im/dataset_train/6

!mv /content/valid_im/dataset_valid/2/* /content/valid_im/dataset_valid/0/
!mv /content/valid_im/dataset_valid/3/* /content/valid_im/dataset_valid/0/
!mv /content/valid_im/dataset_valid/4/* /content/valid_im/dataset_valid/0/
!mv /content/valid_im/dataset_valid/5/* /content/valid_im/dataset_valid/0/
!mv /content/valid_im/dataset_valid/6/* /content/valid_im/dataset_valid/0/

!rm -rf /content/valid_im/dataset_valid/2
!rm -rf /content/valid_im/dataset_valid/3
!rm -rf /content/valid_im/dataset_valid/4
!rm -rf /content/valid_im/dataset_valid/5
!rm -rf /content/valid_im/dataset_valid/6

# obtain data
from torchvision import datasets, transforms
import warnings

def get_one_loader(path=PATH_TRAIN, batch_size=1, num_workers=1):
    """ NEW d
    """
    data_transform = transforms.Compose([transforms.Resize(224, 224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.ImageFolder(path, transform=data_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
    return loader

# test if works
train_loader = get_one_loader(path=PATH_VALID)
classes = ['0: Neutral', '1: Happiness', 
           '2: Sadness', '3: Surprise', 
           '4: Fear', '5: Disgust', '6: Anger']

k = 0
for images, labels in train_loader:
    # since batch_size = 1, there is only 1 image in `images`
    image = images[0]
    # place the colour channel at the end, instead of at the beginning
    img = np.transpose(image, [1,2,0])
    # normalize pixel intensity values to [0, 1]
    img = img / 2 + 0.5
    plt.subplot(2, 3, k+1)
    plt.axis('off')
    plt.imshow(img)
    plt.title(classes[labels[0]])

    k += 1
    if k >= 6:
        break

"""##Helper functions"""

###############################################################################
# Training
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

###############################################################################
# Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def get_accuracy(model, data, train=False):
    with torch.no_grad():
        err = 0
        total = 0
        #for imgs, labels in torch.utils.data.DataLoader(data, batch_size=64):
        for imgs, labels in data: 
            #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################
            
            
            output = model(imgs)
            
            #select index with maximum prediction score
            corr = (output > 0.0).squeeze().long() != labels
            err += int(corr.sum())
            total += imgs.shape[0]

            del imgs
            del labels
        return (total - err) / total

def plot_training_curve(path):
    import matplotlib.pyplot as plt

    train_losses = np.loadtxt("{}_train_loss.csv".format(path))
    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_losses = np.loadtxt("{}_val_loss.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))

    plt.title("Training Curve")

    n = len(train_losses)
    m = len(val_losses)
    # print(n, m)
    plt.plot(range(1, n + 1), train_losses, label="Train")
    plt.plot(range(1, m + 1), val_losses, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(range(1, n + 1), train_acc, label="Train")
    plt.plot(range(1, m + 1), val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    print("Final Training Loss: {}".format(train_losses[-1]))
    print("Final Validation Loss: {}".format(val_losses[-1]))
    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
def enable_dropout(model):
    model.train()
    # for m in model.modules():
    #     if m.__class__.name.startswith('Dropout'):
    #         m.train()

"""# ***Model Training***"""

def train(model, batch_size=64, learning_rate=0.01, num_epochs=1):
    s_time = time.time()
    train_feature = get_one_loader(PATH_TRAIN,batch_size)
    val_feature = get_one_loader(PATH_VALID,batch_size)
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    
    # Obtain the PyTorch data loader objects to load batches of the datasets
    
    
    criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # print(model.parameters())
    iters, viters, train_losses, val_losses, train_acc, val_acc = [], [], [], [], [], []

    # training
    n = 0 # the number of iterations
    m = 0
    for epoch in range(num_epochs):
        train_loss_tot = 0
        val_loss_tot = 0
        train_item = 0
        val_item = 0

        print("Starting epoch ", epoch)
        t_err = 0
        t_total = 0
        v_err = 0
        v_total = 0
        print("\rTrain Progress: {} out of {} = {:05.2f}%. Time passed: {:.2f}ms".format(
            train_item, len(train_feature), train_item * 100. / len(train_feature), 1000. * (time.time() - s_time)), end='')
        enable_dropout(model)

        for features, labels in iter(train_feature):
            #print(labels.shape)
            #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              features = features.cuda()
              labels = labels.cuda()
            #############################################
            
            
            out = model(features)             # forward pass
            loss = criterion(out.flatten(), labels.float()) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            train_loss_tot += loss
            train_item += 1
            iters.append(n)
            n += 1

            corr = (out > 0.0).squeeze().long() != labels
            t_err += int(corr.sum())
            t_total += features.shape[0]

            del labels
            del features
            print("\rTrain Progress: {} out of {} = {:05.2f}%. Time passed: {:.2f}ms current_train_acc: {:05.2f}% ".format(
                train_item, len(train_feature), train_item * 100. / len(train_feature), 1000. * (time.time() - s_time), (t_total-t_err) * 100. / t_total
                ), end='')
        print()
        print("\rValid Progress: {} out of {} = {:05.2f}%. Time passed: {:.2f}ms".format(
            val_item, len(val_feature), val_item * 100. / len(val_feature), 1000. * (time.time() - s_time)), end='')
        with torch.no_grad():      
          for features, labels in iter(val_feature):
            #############################################
              #To Enable GPU Usage
              if use_cuda and torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
              #############################################
              
              out = model(features)             # forward pass
              loss = criterion(out.flatten(), labels.float()) # compute the total loss
              val_loss_tot += loss
              val_item += 1
              viters.append(m)
              m += 1

              corr = (out > 0.0).squeeze().long() != labels
              v_err += int(corr.sum())
              v_total += features.shape[0]
              print("\rValid Progress: {} out of {} = {:05.2f}%. Time passed: {:.2f}ms".format(
                val_item, len(val_feature), val_item * 100. / len(val_feature), 1000. * (time.time() - s_time)), end='')
        print()
              
        # save the training info for every few (1) epoches:
        #iters.append(n)
        train_losses.append(float(train_loss_tot)/train_item)             # compute *average* loss
        #get_acc = get_accuracy(model, train_feature)
        get_acc = (t_total-t_err) * 100. / t_total
        train_acc.append(get_acc) # compute training accuracy 

        val_losses.append(float(val_loss_tot)/val_item)             # compute *average* loss
        get_acc_val = get_accuracy(model, val_feature)
        val_acc.append(get_acc_val)  # compute validation accuracy

        #print("Iter {}: ".format(n))
        print("Train loss: {}, Train acc: {}".format(float(train_loss_tot)/train_item, get_acc))
        print("Validation loss: {}, Validation acc: {}".format(float(val_loss_tot)/val_item, get_acc_val))
  
        # (checkpoint)
        model_path = get_model_name("NE_distinguisher", batch_size, learning_rate, epoch)
        torch.save(model.state_dict(), model_path)  
        # for plotting later (checkpoint)
        np.savetxt("{}_train_loss.csv".format(model_path), train_losses)
        np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
        np.savetxt("{}_val_loss.csv".format(model_path), val_losses)
        np.savetxt("{}_val_acc.csv".format(model_path), val_acc)      
    print("Finished training")

import torchvision.models
googlenet = torchvision.models.alexnet(pretrained=True)
model = googlenet
model.fc = nn.Linear(1024, 1)

class MyAlexNet(nn.Module):

    def __init__(self, num_classes=1):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def my_alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MyAlexNet(**kwargs)
    return model

# training
model = my_alexnet()
use_cuda = True

if use_cuda and torch.cuda.is_available():
  torch.cuda.empty_cache()
  model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')
train(model, batch_size=128, learning_rate=0.0001, num_epochs=15)

model_path = get_model_name("NE_distinguisher", batch_size=128, learning_rate=0.0001, epoch=3)
plot_training_curve(model_path)

model = my_alexnet()
use_cuda = True
if use_cuda and torch.cuda.is_available():
  torch.cuda.empty_cache()
  model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')
model_path = get_model_name("NE_distinguisher", batch_size=128, learning_rate=0.0001, epoch=2)
state = torch.load(model_path)
model.load_state_dict(state)
model.eval()
def check_acc():
  val_feature = get_one_loader(PATH_VALID,1)
  get_acc_val = get_accuracy(model, val_feature)
  print(get_acc_val)
check_acc()

train_loader = get_one_loader(path=PATH_VALID)
classes = ['not HA', 'HA' ]

k = 0
for images, labels in train_loader:
    # since batch_size = 1, there is only 1 image in `images`
    image = images[0]
    if use_cuda and torch.cuda.is_available():
      images = images.cuda()
    output = model(images)
    pred = (output > 0.0).squeeze().long()
    # place the colour channel at the end, instead of at the beginning
    img = np.transpose(image, [1,2,0])
    # normalize pixel intensity values to [0, 1]
    img = img / 2 + 0.5
    plt.subplot(2, 3, k+1)
    plt.axis('off')
    plt.imshow(img)
    plt.title(classes[labels[0]]+'(' + classes[pred] + ')')

    k += 1
    if k >= 6:
        break

from sklearn.metrics import confusion_matrix
batch_size = 1
def plot_confusion_matrix(model, classes,
              normalize=False,
              title=None,
              cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    val_feature = get_one_loader(PATH_VALID,batch_size)
    y_true = []
    y_pred = []
    with torch.no_grad():  
      for features, labels in iter(val_feature):
        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
          features = features.cuda()
        #############################################
        
          
        out = model(features)             # forward pass
        pred = out.max(1, keepdim=True)[1]
        pred = pred.cpu()
        for predit_num in pred.numpy():
          y_pred.append(predit_num[0])
          #print(predit_num[0])
        for actual_num in labels.numpy():
          y_true.append(actual_num)
          #print(actual_num)
        del features
    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

#class_names = ['','NE','HA','SA','SU','AF','DI','AN']
class_names = ['','Not HA','','HA']
plot_confusion_matrix(model, classes=class_names,
                      title='Confusion matrix')
plt.savefig('primary_model_confusion_matrix.png')
plt.show()