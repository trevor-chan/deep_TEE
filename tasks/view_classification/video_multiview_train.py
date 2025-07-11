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
import json

import sys
sys.path.append('../../vision')

import datasets
import training_loop
import losses
import models

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
print(device)


dfs = []
for csvfile in sorted(glob.glob('/home/trevor/repos/deep_TEE/labels/cyclic_view_labeling/*.csv')):
    dfs.append(pd.read_csv(csvfile, index_col=False))

view_labels = pd.concat(dfs).reset_index(drop=True)

# currently only use the positive predictions:
view_labels = view_labels.loc[view_labels['Unnamed: 2'] == 'y'].reset_index(drop=True)

view_classes = np.unique(np.array(view_labels['prediction']))
view_classes

views_dir = '../../data/semi-processed/supervised/growing_multiview_training/'
views = sorted([os.path.basename(view) for view in glob.glob(views_dir + '*')])

view_names = [views[index] for index in view_classes]

print(f'{view_classes} : {view_names}')


class MultiviewDataset(torch.utils.data.Dataset):

    def __init__(self, labels_df, transform=None, duration=32, skip=1):
        self.labels_df = labels_df
        self.transform = transform
        self.duration = duration
        self.skip = skip
        self.view_classes = torch.unique(torch.tensor(view_labels['prediction']))

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):      
        label = torch.argwhere(self.view_classes == int(self.labels_df.iloc[idx]['prediction']))
        label = torch.nn.functional.one_hot(label, num_classes=len(self.view_classes))
        
        data = self.__read_images(self.labels_df.iloc[idx]['clip_path'], self.duration, self.skip)
        
        if self.transform:
            data = self.transform(data)
            
        data = data.permute(1,0,2,3) # reshape to (C,T,H,W)
            
        return {'data':data, 'label':label, 'path':self.labels_df.iloc[idx]['clip_path']}
    
    
    def __read_images(self, path, duration=32, skip=1):
        frame_paths = glob.glob(path+'/*')

        if len(frame_paths) <= duration * skip:
            start = 0
            stop = len(frame_paths)
            pad = (duration * skip - stop) // skip
        else:
            start = np.random.randint(0, len(frame_paths) - duration * skip + 1)
            stop = start + duration * skip
            pad = 0
        frames = []
        for frame in sorted(frame_paths)[start:stop:skip]:
            frames.append(torchvision.transforms.functional.to_tensor((Image.open(frame))))
        for i in range(pad):
            frames.append(torch.zeros_like(frames[-1]))
        frames = torch.stack(frames, dim=0)
        return frames
    
# plt.imshow(train_dataset[0]['data'][:,0].permute(1,2,0))



# train_transforms = datasets.TransformsComposed('train', size=(224,224))
train_transforms = datasets.TransformsComposed('valtest', size=(224,224))
valtest_transforms = datasets.TransformsComposed('valtest', size=(224,224))

view_labels = view_labels.sample(frac=1, random_state=0).reset_index(drop=True)

train_dataset = MultiviewDataset(view_labels[:3000], transform=train_transforms, duration=16, skip=1)
test_dataset = MultiviewDataset(view_labels[3000:], transform=valtest_transforms, duration=16, skip=1)

class SimpleLoss:
    def __init__(self,):
        self.crossentropy = torch.nn.CrossEntropyLoss()
        self.output_activation = torch.nn.Softmax(dim=1)

    def __call__(self, net, data, label):
        x = data
        y = net(x).squeeze()
        y = self.output_activation(y)
        label = label.squeeze()
        assert y.shape == label.shape, "mismatched dimensions in model output"
        loss = self.crossentropy(y, label)
        return loss
    
loss_fn = SimpleLoss()


model = models.fetch_s3d_model(
    num_classes = len(view_classes),
    dropout = 0.2,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,)


import torch
import numpy as np
import os
import time
import copy
import json
import pickle
import datasets
import matplotlib.pyplot as plt

def collate_fn(batch):
  return {
      'data': torch.stack([x['data'] for x in batch], dim=0),
      'label': torch.stack([x['label'] for x in batch], dim=0),
}

def simple_training_loop(
    run_dir             = '.',      # Output directory.
    dataset             = None,     # Training dataset
    validation_dataset  = None,     # Validation dataset
    network             = None,     # Options for model and preconditioning.
    loss                = None,     # Options for loss function.
    optimizer           = None,     # Options for optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = 32,       # Total batch size for one training iteration.
    num_workers         = 16
    ,        # Number of data loading workers.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    device              = torch.device('cuda'),
    kimg_per_tick       = 100,      # How often to log training and validation stats.
    kimg_per_save       = 1000,     # How often to save the training state.
    val_length          = 100,       # How many data parcels to run validation on (more takes longer)
):
    # Initialize.
    start_time = time.time()
    if os.path.isdir(run_dir):
        # raise RuntimeError(f'Output directory "{run_dir}" already exists')
        os.makedirs(run_dir, exist_ok=True)
    else:
        os.makedirs(run_dir, exist_ok=True)

    # Load dataset.
    print('Loading dataset...')

    dataset_sampler = iter(datasets.InfiniteSampler(dataset=dataset, rank=0, num_replicas=1, shuffle=True, seed=seed, window_size=0.5))
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=dataset_sampler, collate_fn=collate_fn))
    if validation_dataset:
        val_dataset = validation_dataset
        val_dataset_sampler = iter(datasets.InfiniteSampler(dataset=val_dataset, rank=0, num_replicas=1, shuffle=True, seed=seed, window_size=0.5))
        val_dataset_iterator = iter(torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=1, sampler=val_dataset_sampler, collate_fn=collate_fn))

    # Construct network.
    print('Constructing network...')
    net = network
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    print('Setting up optimizer...')
    loss_fn = loss
    optimizer = optimizer
    
    # Setup logs.
    print('Setting up logs...')
    running_loss = []
    training_stats = dict(
        tick = [],
        tick_start_time = [],
        tick_runtime = [],
        loss = [],
        std = [],
        valloss = [],
        valstd = [],
    )

    # Train.
    print(f'Training for {total_kimg} kimg...')
    print()
    cur_nimg = 0
    cur_tick = 0
    cur_save = 1
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    # dist.update_progress(cur_nimg // 1000, total_kimg)
    
    while True:
        
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        parcel = next(dataset_iterator)
        data = parcel['data']
        label = parcel['label']
        data = data.to(device).to(torch.float32)
        label = label.to(device).to(torch.float32)
        loss = loss_fn(net=net, data=data, label=label)
        running_loss.append(loss)
        loss.sum().backward()
        
        
        # print(f'data : {data.shape}')
        # print(f'label : {label}')
        # print(f'loss : {loss}')
        # print(f'image : {parcel["acc"]}')
        # if cur_nimg % 100 == 0:
        #     plt.imshow(data[0,:,0,:,:].permute(1,2,0).cpu())

        # Update weights.
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Perform maintenance tasks once per tick.
        cur_tick += 1
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        
        # Save full dump of the training state.
        if (not done) and (cur_tick != 0) and (cur_nimg > (cur_save) * kimg_per_save * 1000):
            cur_save += 1
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
            print(f'saved to training-state-{cur_nimg//1000:06d}.pt')
        
        # Validation
        if validation_dataset:
            model.eval()
            with torch.no_grad():
                val_loss = []
                for idx in range(min(val_length, len(val_dataset))):
                    parcel = next(val_dataset_iterator)
                    data = parcel['data']
                    label = parcel['label']
                    data = data.to(device).to(torch.float32)
                    label = label.to(device).to(torch.float32)
                    loss = loss_fn(net=net, data=data, label=label)
                    val_loss.append(loss)
                training_stats['valloss'].append(torch.mean(torch.stack(val_loss)).item())
                training_stats['valstd'].append(torch.std(torch.stack(val_loss)).item())
            model.train()
                
        # Update logs.
        training_stats['tick'].append(cur_tick)
        training_stats['tick_start_time'].append(tick_start_time)
        training_stats['tick_runtime'].append(time.time() - tick_start_time)
        training_stats['loss'].append(torch.mean(torch.stack(running_loss)).item())
        training_stats['std'].append(torch.std(torch.stack(running_loss)).item())
        json.dump(training_stats, open(os.path.join(run_dir, 'log.json'), 'wt'))
        running_loss = []
        if validation_dataset:
            print(f"tick {cur_tick}, loss {training_stats['loss'][-1]:.4f}, std {training_stats['std'][-1]:.4f}, valloss {training_stats['valloss'][-1]:.4f}, valstd {training_stats['valstd'][-1]:.4f}, time {training_stats['tick_runtime'][-1]:.2f} sec")
        else:
            print(f"tick {cur_tick}, loss {training_stats['loss'][-1]:.4f}, std {training_stats['std'][-1]:.4f}, time {training_stats['tick_runtime'][-1]:.2f} sec")

        # Update state.
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        if done:
            break

    # Done.
    print()
    print('Exiting...')

#----------------------------------------------------------------------------



simple_training_loop(
    run_dir             = '../../experiments/multiview_classification_s3d_224_16_1_test1',  # Output directory.
    dataset             = train_dataset,                # Options for training set.
    validation_dataset  = test_dataset,                 # Options for training set.
    network             = model,                        # Options for model and preconditioning.
    loss                = loss_fn,                      # Options for loss function.
    optimizer           = optimizer,                    # Options for optimizer.
    seed                = 0,                            # Global random seed.
    batch_size          = 32,                           # Total batch size for one training iteration.
    num_workers         = 4,                            # Number of data loading workers.
    total_kimg          = 1000,                          # Training duration, measured in thousands of training images.
    kimg_per_tick       = 25,                           # How often to log training and validation stats.
    kimg_per_save       = 25,                           # How often to save the training state.
    device              = torch.device('cuda'),
    val_length          = 1000,       # How many data parcels to run validation on (more takes longer)
)