import sys

sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import os
import random
from natsort import natsorted, index_natsorted

import torch
import torchvision
from PIL import Image
import time
import copy
from tqdm import tqdm
import torchvision.models as models

import sys

import wandb   


from torch.utils.data import WeightedRandomSampler

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

# from multi_to_binary import multi_to_binary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

iteration = '01'

# parameters
path = '../../data/semi-processed/supervised/multiview'
seed = 42
batch_size = 64


training_dir = f'../../data/semi-processed/supervised/growing_multiview_training'

mask = False
if mask:
    masked = '_mask'
else:
    masked = ''

print(f'Using {training_dir} for training')

run = wandb.init(
    project="echo_view_classification",
)

wandb.config = {
    "batch_size": batch_size,
}



# Specifying data augmentation transforms

class transforms_composed:
    """Rotate by one of the given angles."""

    def __init__(self, mode, apply=True, mask=False):
        self.mode = mode
        self.mask=False
        self.size = (224,224)
        if mask:
            self.mask = True
            mask = Image.open('../assets/dicom-mask.png')
            mask.load()
            mask = mask.resize(self.size,Image.NEAREST)
            masking_array = (np.array(mask)[:,:,3])
            masking_array[masking_array!=0]=1
            self.masking_array = (masking_array.astype(float)-1)*-1
        self.apply = apply
        self.randomResizedCrop = torchvision.transforms.RandomResizedCrop((self.size[1],self.size[0]), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), antialias=True)
        self.randomRotate = torchvision.transforms.RandomRotation(30)
        self.xflip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.resize = torchvision.transforms.Resize((self.size[1],self.size[0]), antialias=True)

    def __call__(self, x):
        tensorconvert=torchvision.transforms.ToTensor()
        if self.mode=='train' and self.apply:
            x = tensorconvert(x)
            if self.mask:
                x = x*self.masking_array
            if random.random()>0.15:
                x=torchvision.transforms.functional.adjust_brightness(x,np.amax((0.1,1+np.random.randn()*0.25)))
            if random.random()>0.15:
                x=torchvision.transforms.functional.adjust_contrast(x,np.amax((0.1,1+np.random.randn()*0.25)))
            if random.random()>0.15:
                x = self.randomResizedCrop(x)
                x = self.randomRotate(x)
                x=self.xflip(x)
            else:
                x = self.resize(x)
            # x = torch.unsqueeze(x[0,:,:],0)
        else:
            x = tensorconvert(x)
            x = self.resize(x)
            if self.mask:
                x = x*self.masking_array
            # x = torch.unsqueeze(x[0,:,:],0)
        return x.type(torch.float)

train_transforms = transforms_composed('train', mask=mask)
valtest_transforms = transforms_composed('valtest', mask=mask)


# load the data sets for multiclass classification model

data_dir = f'{path}'

train_dataset = torchvision.datasets.ImageFolder(training_dir, train_transforms)
val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'),valtest_transforms)
test_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'),valtest_transforms)

class_names = train_dataset.classes


# weighted random sampling for imbalanced classes:
class_sample_count = np.array([len(os.listdir(training_dir+'/'+name)) for name in class_names])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in train_dataset.targets])

samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler = sampler)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

dataloaders = {'train':train_dataloader,'val':val_dataloader,'test':test_dataloader}
# dataloaders_no_aug = {'train':train_dataloader_no_transforms,'val':val_dataloader,'test':test_dataloader}

print(class_names)
print(len(class_names))


def train_model(model, loss_fn, optimizer, dataloaders, num_epochs=25, experiment_dir = None):
    if experiment_dir is None:
        experiment_dir = f'./'
    else:
        os.makedirs(experiment_dir, exist_ok=True)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                wandb.log({"loss": epoch_loss,})
            elif phase == 'val':
                wandb.log({"val_loss": epoch_loss,})
                torch.save(model.state_dict(), f'{experiment_dir}/model_weights_{str(epoch).zfill(3)}.pt')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'{experiment_dir}/model_weights_best.pt')
    return model


#train multiclass model (2 days)

#EfficientNet V2
model = models.efficientnet_v2_l(num_classes=len(class_names), dropout=0.2).to(device)  # Biggo model

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

model_ft = train_model(model, loss_fn, optimizer, dataloaders, num_epochs=25, experiment_dir = '../../experiments/multiview_classification_02')