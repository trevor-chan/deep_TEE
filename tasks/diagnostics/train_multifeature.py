import torch
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import torchvision
import tqdm
import glob
from PIL import Image
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('../../vision')

import datasets
import training_loop
import losses
import models
import optimizers

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)




labels = pd.read_csv("../../labels/semi_cleaned_labels_3_features_2024-10-21.csv")
labels = labels.drop(columns=['MRN','SURGERY_DATE'])

new_labels = labels.replace('nan', np.nan)
new_labels = new_labels.replace('nc', np.nan)

# drop nans:
new_labels.dropna(inplace=True) # needs to happen before sorting into pre and post

new_labels_pre = new_labels[['ACC_NUM','LVEF_pre','RVSF_pre','TR_pre']]
new_labels_pre.columns = ['ACC_NUM','LVEF','RVSF','TR']
new_labels_pre['prepost'] = 0

new_labels_post = new_labels[['ACC_NUM','LVEF_post','RVSF_post','TR_post']]
new_labels_post.columns = ['ACC_NUM','LVEF','RVSF','TR']
new_labels_post['prepost'] = 1

new_labels = pd.concat([new_labels_pre, new_labels_post])
new_labels.reset_index(drop=True, inplace=True)

new_labels['LVEF'] = new_labels['LVEF'].astype(float)/100
new_labels['RVSF'] = new_labels['RVSF'].astype(float)/6
new_labels['TR'] = new_labels['TR'].astype(float)/7

ground_truths = pd.read_csv("../../labels/500_reports_groundtruth.csv")['ACC_NUM']

# drop nans:
new_labels.dropna(inplace=True)

ground_truths = list(ground_truths.astype(float))

training_labels = new_labels.drop(new_labels[new_labels['ACC_NUM'].isin(ground_truths)].index)
evaluation_labels = new_labels.drop(new_labels[~new_labels['ACC_NUM'].isin(ground_truths)].index) # not quite correct, but probably good enough until we clean the gt dataset

print(len(training_labels), len(evaluation_labels))


view_prediction_csvs = glob.glob("../../labels/view_predictions/cycle_*/predictions_all_cycle_*.csv", recursive=True)
predictions = []
for cycle_preds in view_prediction_csvs:
    cycle = pd.read_csv(cycle_preds)
    # predictions.append(cycle.loc[cycle['prediction'].isin([3,4,5,6,7,8,9])]) # If filtering by view - currently all 2,4,5 chamber with/without color
    predictions.append(cycle) # If not filtering by view

all_cycles = pd.concat(predictions)
all_cycles = all_cycles.reset_index()
val_cycles = all_cycles.sample(frac=0.01, random_state=0)
train_cycles = all_cycles.drop(val_cycles.index)

val_cycles = val_cycles.reset_index()
train_cycles = train_cycles.reset_index()


print(len(train_cycles))
removals = []
for i in tqdm.tqdm(range(len(train_cycles['clip_path']))):
    try:
        if int(os.path.basename(os.path.split(train_cycles['clip_path'].iloc[i])[0])) not in list(training_labels['ACC_NUM']):
        # if int(os.path.basename(os.path.split(train_cycles['clip_path'].iloc[i])[0])) in ground_truths:
            removals.append(i)
    except:
        removals.append(i)
train_cycles = train_cycles.drop(removals)
print(len(train_cycles))

print(len(val_cycles))
removals = []
for i in tqdm.tqdm(range(len(val_cycles['clip_path']))):
    try:
        if int(os.path.basename(os.path.split(val_cycles['clip_path'].iloc[i])[0])) not in list(training_labels['ACC_NUM']):
        # if int(os.path.basename(os.path.split(val_cycles['clip_path'].iloc[i])[0])) in ground_truths:
            removals.append(i)
    except:
        removals.append(i)
        
val_cycles = val_cycles.drop(removals)
print(len(val_cycles))
train_cycles = train_cycles.reset_index(drop=True)
val_cycles = val_cycles.reset_index(drop=True)


train_clips = list(train_cycles['clip_path'])
val_clips = list(val_cycles['clip_path'])

# train_clips = ['/data/deep_tee/' + clip[8:] for clip in train_clips]
# val_clips = ['/data/deep_tee/' + clip[8:] for clip in val_clips]



prepost_pred = pd.read_csv('/home/trevor/repos/deep_TEE/labels/preop_postop_prediction.csv')
acc_nums = [prepost_pred['clip_path'][i].split('/')[0] if type(prepost_pred['clip_path'][i])==str else '' for i in range(len(prepost_pred['clip_path']))]
dcm_nums = [prepost_pred['clip_path'][i].split('/')[-1] if type(prepost_pred['clip_path'][i])==str else ''  for i in range(len(prepost_pred['clip_path']))]
prepost_pred['ACC_NUM'] = acc_nums
prepost_pred['DCM_NUM'] = dcm_nums

# train_transforms = datasets.TransformsComposed('train', size=(224,224))
# valtest_transforms = datasets.TransformsComposed('valtest', size=(224,224))

train_transforms = datasets.TransformsComposed('train', size=(128,128))
valtest_transforms = datasets.TransformsComposed('valtest', size=(128,128))

features = ['LVEF','RVSF','TR']


train_dataset = datasets.VideoDataset(clip_paths=train_clips, labels_df=training_labels, features=features, time_df=prepost_pred, transform=train_transforms, duration=32, skip=2)
test_dataset = datasets.VideoDataset(clip_paths=val_clips, labels_df=training_labels,  features=features, time_df=prepost_pred, transform=valtest_transforms, duration=32, skip=2)


# model = models.fetch_s3d_model(
#     num_classes = 3,
#     dropout = 0,
# ).to(device)

model = models.fetch_vit_model(
        spatial_size = (128,128),
        temporal_size = 32,
        residual_pool = False,
        residual_with_cls_embed = False,
        stochastic_depth_prob = 0.0,
        rel_pos_embed = True,
        proj_after_attn = True,
        num_classes = 3,
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.000005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,)

loss_fn = losses.MaskedLoss()

training_loop.simple_training_loop(
    run_dir             = '../experiments/test_vit_128_32_1_0_3feature_anyview_test1',  # Output directory.
    # run_dir             = '../experiments/test_s3d_224_32_1_0_3feature_anyview_test1',  # Output directory.
    dataset             = train_dataset,                # Options for training set.
    validation_dataset  = test_dataset,                 # Options for training set.
    network             = model,                        # Options for model and preconditioning.
    loss                = loss_fn,                      # Options for loss function.
    optimizer           = optimizer,                    # Options for optimizer.
    seed                = 0,                            # Global random seed.
    batch_size          = 32,                           # Total batch size for one training iteration.
    num_workers         = 16,                            # Number of data loading workers.
    total_kimg          = 100000,                          # Training duration, measured in thousands of training images.
    kimg_per_tick       = 20,                           # How often to log training and validation stats.
    kimg_per_save       = 500,                           # How often to save the training state.
    device              = torch.device('cuda'),
)