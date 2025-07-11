import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import os
import random
from natsort import natsorted, index_natsorted
from statistics import mode
import glob

import torch
import torchvision
from PIL import Image
import time
import copy
import tqdm
import re
import torchvision.models as models


from torch.utils.data import WeightedRandomSampler

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_grad_enabled(False)

print('running inference')

path = '/home/trevor/repos/deep_TEE/data/semi-processed/unsupervised/train'
model_path = '/home/trevor/repos/deep_TEE/experiments/multiview_classification_01/model_weights_010.pt'

for i in range(1, 70):

    cycle = 100 * i
    start = 100 * (i - 1)
    stop = 100 * i
    save_snapshots = False


    # parameters
    seed = 42

    # Specifying data augmentation transforms

    class transforms_composed:
        """Rotate by one of the given angles."""

        def __init__(self, mode, apply=True, mask=False):
            self.mode = mode
            self.mask=False
            self.size = (200,150)
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
                # x = torch.unsqueeze(x[0,:,:],0)
            else:
                x = tensorconvert(x)
                if self.mask:
                    x = x*self.masking_array
                # x = torch.unsqueeze(x[0,:,:],0)
            return x.type(torch.float)

    valtest_transforms = transforms_composed('valtest')


    class inference_dataset(torch.utils.data.Dataset):
        def __init__(self, path, transform=None, start=None, stop=None):
            
            print('Initializing inference dataset')
            self.path = path
            
            studies = glob.glob(path+'/*')
            self.studies = []
            self.clips = []
            self.imgs = []
            for study in tqdm.tqdm(natsorted(studies)[start:stop]):
                for clip in natsorted(glob.glob(study+'/*')):
                    img_list = glob.glob(clip+'/*.png', recursive=True)
                    self.imgs = self.imgs + img_list
                    self.clips = self.clips + [clip for i in range(len(img_list))]
                    self.studies = self.studies + [study for i in range(len(img_list))]

            self.transform = transform

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = self.imgs[idx]
            
            image = np.array(Image.open(img_name))

            if self.transform is not None:
                image = self.transform(image)

            return image, self.clips[idx], self.studies[idx]

    class_names = ['deep_tg_col_a', 'dta_sax', 'dta_sax_arch', 'me_2_ch', 'me_2_ch_col_m', 'me_4_ch', 'me_4_ch_col_m', 'me_4_ch_col_t', 'me_5_ch', 'me_5_ch_col', 'me_av_sax', 'me_av_sax_col', 'me_bicaval', 'me_bicaval_col', 'me_in_out', 'me_in_out_col_p', 'me_in_out_col_t', 'me_lax', 'me_lax_col_a', 'me_lax_col_m', 'me_lax_col_mv_av', 'me_mit_com', 'me_mit_com_col_m', 'other', 'tg_basal_sax', 'tg_mid_pap_sax', 'ue_asc_ao_sax']


    print('Made directory for cycle_{}'.format(str(cycle).zfill(2)))
    os.makedirs(f'/home/trevor/repos/deep_TEE/labels/view_predictions/cycle_{str(cycle).zfill(2)}', exist_ok=True)

    # running in cycles of 200 studies
    inference = inference_dataset(path, transform=valtest_transforms, start=start, stop=stop)


    # Creating the model
    print('Building model')
    # efnet_model = models.efficientnet_v2_m(num_classes=len(class_names), dropout=0.2)  # medium model
    efnet_model = models.efficientnet_v2_l(num_classes=len(class_names), dropout=0.2)  # large model


    efnet_model = efnet_model.to(device)
    efnet_model.eval()
    efnet_model.load_state_dict(torch.load(model_path, map_location=device))
    multiclass_model = efnet_model


    def evaluate_multiclass_model_clipwise(model, test_dataset, increment=10, topk=1):
        was_training = model.training
        model.eval()

        predictions = []
        clip_labels = []
        subjet_labels = []
        
        with torch.no_grad():
            for i, inputs in tqdm.tqdm(enumerate(test_dataset), total=len(test_dataset)):
                if i%increment!=0:
                    continue
                clip_labels.append(inputs[1])
                subjet_labels.append(inputs[2])
                
                image = torch.unsqueeze(inputs[0], dim=0).to(device)
                outputs = model(image)
                predictions.append(outputs[0,:].cpu().numpy())

        aggregate_predictions = []
        aggregate_labels = []
        running_sum = np.zeros(len(class_names))
        current_clip = clip_labels[0]
        for clip_label in clip_labels:
            if current_clip != clip_label:
                aggregate_predictions.append(running_sum)
                aggregate_labels.append(current_clip)
                running_sum = np.zeros(len(class_names))
                current_clip = clip_label
            else:
                running_sum += predictions[clip_labels.index(clip_label)]
            
        if topk == 1:
            test_labels_predicted = np.argmax(aggregate_predictions,axis=1)
        else:
            test_labels_predicted = np.argsort(aggregate_predictions,axis=1)[:,-topk:]
        
        return np.array(aggregate_predictions), np.array(test_labels_predicted), aggregate_labels


    # Evaluate multiclass model

    def class_2_index(class_name, class_names):
        indices = np.where(np.array(class_names)==class_name)
        return np.eye(len(class_names))[indices]

    print('Evaluating')
    test_predictions, test_labels, clip_paths = evaluate_multiclass_model_clipwise(multiclass_model, inference, increment=5, topk=1)

    results = pd.DataFrame({'clip_path':clip_paths, 'prediction':test_labels})
    results.to_csv(f'/home/trevor/repos/deep_TEE/labels/view_predictions/cycle_{str(cycle).zfill(2)}/predictions_all_cycle_{str(cycle).zfill(2)}.csv', index=False)


    if not save_snapshots:
        # quit()
        continue

    rows = 10
    cols = 10
    print('Saving image snapshots')

    for id, view in tqdm.tqdm(enumerate(class_names)):
        print(view)
        for page in range(len(results.loc[results['prediction']==id]['clip_path'])//rows + 1):
            fig, ax = plt.subplots(rows, cols, figsize=(40,rows * 3))
            for i, clip_path in enumerate(list(results.loc[results['prediction']==id]['clip_path'])[page * rows:]):
                images = glob.glob(clip_path+'/*.png')

                indices = sorted(list(torch.randint(len(images), size=(cols,))))
                indices = indices
                for j, index in enumerate(indices):
                    ax[i, j].imshow(np.array(Image.open(images[index])))
                    ax[i, j].axis('off')
                ax[i, 2].set_title(f'\n{clip_path}')
                if i == rows-1:
                    break
            fig.suptitle(f'{view}  page {page}', y=.92)
            plt.savefig(f'/home/trevor/repos/deep_TEE/labels/view_predictions/cycle_{str(cycle).zfill(2)}/{view}_page_{page}_cycle_{str(cycle).zfill(2)}.png')
            plt.close()
        results.loc[results['prediction']==id].to_csv(f'/home/trevor/repos/deep_TEE/labels/view_predictions/cycle_{str(cycle).zfill(2)}/{view}_cycle_{str(cycle).zfill(2)}.csv', index=False)
