import torch
import numpy as np
import os
import time
import copy
import json
import pickle
import datasets
import matplotlib.pyplot as plt


def simple_training_loop(
    run_dir             = '.',      # Output directory.
    dataset             = None,     # Training dataset
    validation_dataset  = None,     # Validation dataset
    network             = None,     # Options for model and preconditioning.
    loss                = None,     # Options for loss function.
    optimizer           = None,     # Options for optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = 32,       # Total batch size for one training iteration.
    num_workers         = 8,        # Number of data loading workers.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    device              = torch.device('cuda'),
    kimg_per_tick       = 100,      # How often to log training and validation stats.
    kimg_per_save       = 1000,     # How often to save the training state.
    val_length          = 200,       # How many data parcels to run validation on (more takes longer)
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
    dataset_obj = dataset
    dataset_sampler = iter(datasets.InfiniteSampler(dataset=dataset_obj, rank=0, num_replicas=1, shuffle=True, seed=seed, window_size=0.5))
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=batch_size, num_workers=num_workers, sampler=dataset_sampler, collate_fn=datasets.collate_fn))
    if validation_dataset:
        val_dataset_obj = validation_dataset
        val_dataset_sampler = iter(datasets.InfiniteSampler(dataset=val_dataset_obj, rank=0, num_replicas=1, shuffle=True, seed=seed, window_size=0.5))
        val_dataset_iterator = iter(torch.utils.data.DataLoader(dataset=val_dataset_obj, batch_size=batch_size, num_workers=1, sampler=val_dataset_sampler, collate_fn=datasets.collate_fn))


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
                net.eval()
                with torch.no_grad():
                    val_loss = []
                    for idx in range(val_length):
                        parcel = next(val_dataset_iterator)
                        data = parcel['data']
                        label = parcel['label']
                        data = data.to(device).to(torch.float32)
                        label = label.to(device).to(torch.float32)
                        loss = loss_fn(net=net, data=data, label=label)
                        val_loss.append(loss)
                    training_stats['valloss'].append(torch.mean(torch.stack(val_loss)).item())
                    training_stats['valstd'].append(torch.std(torch.stack(val_loss)).item())
                net.train()
        else:
            training_stats['valloss'].append(float('nan'))
            training_stats['valstd'].append(float('nan'))
                
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