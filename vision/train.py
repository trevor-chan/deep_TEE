import torch
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import torchvision
import tqdm

import datasets
import training_loop
import losses
import models

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f'using device {device}')

ef_reports = pd.read_csv("../labels/EF_reports.csv")
ef_reports = ef_reports.drop(columns=['MRN','SURGERY_DATE','JOINED_IMPRESSION','EF_post'])
ef_reports = ef_reports.dropna()

cycle1000 = pd.read_csv("../data/feedback_labeling/cycle_1000/predictions_all_cycle_1000.csv")
cycle2000 = pd.read_csv("../data/feedback_labeling/cycle_2000/predictions_all_cycle_2000.csv")
cycle3000 = pd.read_csv("../data/feedback_labeling/cycle_3000/predictions_all_cycle_3000.csv")
cycle4000 = pd.read_csv("../data/feedback_labeling/cycle_4000/predictions_all_cycle_4000.csv")
cycle5000 = pd.read_csv("../data/feedback_labeling/cycle_5000/predictions_all_cycle_5000.csv")
cycle1000 = cycle1000.loc[cycle1000['prediction']==1]
cycle2000 = cycle2000.loc[cycle2000['prediction']==1]
cycle3000 = cycle3000.loc[cycle3000['prediction']==1]
cycle4000 = cycle4000.loc[cycle4000['prediction']==1]
cycle5000 = cycle5000.loc[cycle5000['prediction']==1]

train_cycles = pd.concat((cycle1000, cycle3000, cycle4000, cycle4000,))

test_cycles = cycle2000

# ef_reports = ef_reports.reset_index()
train_cycles = train_cycles.reset_index()
test_cycles = test_cycles.reset_index()

print(len(train_cycles))
removals = []
for i in range(len(train_cycles['clip_path'])):
    if int(os.path.basename(os.path.split(train_cycles['clip_path'].iloc[i])[0])) not in list(ef_reports['ACC_NUM']):
        removals.append(i)
# for r in removals[::-1]:
#     train_cycles = train_cycles.drop(r)
train_cycles = train_cycles.drop(removals)
print(len(train_cycles))

print(len(test_cycles))
removals = []
for i in range(len(test_cycles['clip_path'])):
    if int(os.path.basename(os.path.split(test_cycles['clip_path'].iloc[i])[0])) not in list(ef_reports['ACC_NUM']):
        removals.append(i)
# for r in removals[::-1]:
#     test_cycles = test_cycles.drop(r)
test_cycles = test_cycles.drop(removals)
print(len(test_cycles))

train_cycles = train_cycles.reset_index()
test_cycles = test_cycles.reset_index()

train_clips = train_cycles['clip_path']
test_clips = test_cycles['clip_path']

# train_transforms = datasets.TransformsComposed('train', size=(64,64))
# valtest_transforms = datasets.TransformsComposed('valtest', size=(64,64))
train_transforms = datasets.TransformsComposed('train', size=(224,224))
valtest_transforms = datasets.TransformsComposed('valtest', size=(224,224))

train_dataset = datasets.VideoDataset(clip_paths=train_clips, labels_df=ef_reports, transform=train_transforms, duration=16, skip=2)
test_dataset = datasets.VideoDataset(clip_paths=test_clips, labels_df=ef_reports, transform=valtest_transforms, duration=16, skip=2)

model = models.fetch_s3d_model(
    num_classes = 1,
    dropout = 0.2,
).to(device)

# model = models.fetch_vit_model(
#         spatial_size = (64,64),
#         temporal_size = 64,
#         residual_pool = False,
#         residual_with_cls_embed = False,
#         stochastic_depth_prob = 0.0,
#         rel_pos_embed = True,
#         proj_after_attn = True,
#         num_classes = 1,
#     ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,)

loss_fn = losses.SimpleLoss()

training_loop.simple_training_loop(
    run_dir             = '../experiments/s3d_224_16_2_0',  # Output directory.
    dataset             = train_dataset,                # Options for training set.
    validation_dataset  = test_dataset,                 # Options for training set.
    network             = model,                        # Options for model and preconditioning.
    loss                = loss_fn,                      # Options for loss function.
    optimizer           = optimizer,                    # Options for optimizer.
    seed                = 0,                            # Global random seed.
    batch_size          = 32,                           # Total batch size for one training iteration.
    num_workers         = 8,                            # Number of data loading workers.
    total_kimg          = 20000,                        # Training duration, measured in thousands of training images.
    kimg_per_tick       = 10,                           # How often to log training and validation stats.
    kimg_per_save       = 50,                           # How often to save the training state.
    device              = torch.device('cuda'),
)

