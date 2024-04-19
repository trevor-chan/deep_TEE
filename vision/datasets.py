import numpy as np
import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

def collate_fn(batch):
  return {
      'data': torch.stack([x['data'] for x in batch], dim=0),
      'label': torch.tensor([x['label'] for x in batch]),
      'acc': [x['acc'] for x in batch],
}

class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, clip_paths, labels_df, transform=None, duration=32, skip=1):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.clip_paths = clip_paths
        self.duration = duration
        self.skip = skip

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):  
        acc_num = os.path.basename(os.path.split(self.clip_paths[idx])[0])
        try:
            self.labels_df.loc[self.labels_df['ACC_NUM']==int(acc_num)]['EF_pre'].item()
        except:
            print(idx)
            print(acc_num)
            print(self.labels_df.loc[self.labels_df['ACC_NUM']==int(acc_num)])
        label = self.labels_df.loc[self.labels_df['ACC_NUM']==int(acc_num)]['EF_pre'].item()
        
        label = label / 100
        
        data = self.__read_images(self.clip_paths[idx], self.duration, self.skip)
        
        if self.transform:
            data = self.transform(data)
            
        data = data.permute(1,0,2,3) # reshape to (C,T,H,W)
            
        return {'data':data, 'label':label, 'acc':acc_num}
    
    
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


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
            
            
class TransformsComposed:
    def __init__(self, mode, apply=True, mask=False, size=(150,200)):
        self.mode = mode
        self.mask=False
        self.size = (size[1],size[0])
        if mask:
            self.mask = True
            mask = Image.open('../assets/dicom-mask.png')
            mask.load()
            mask = mask.resize(self.size,Image.NEAREST)
            masking_array = (np.array(mask)[:,:,3])
            masking_array[masking_array!=0]=1
            self.masking_array = (masking_array.astype(float)-1)*-1
        self.apply = apply
        self.resize = torchvision.transforms.v2.Resize((self.size[1],self.size[0]), antialias=True)
        self.randomResizedCrop = torchvision.transforms.v2.RandomResizedCrop((self.size[1],self.size[0]), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), antialias=True)
        self.randomRotate = torchvision.transforms.v2.RandomRotation(30)
        self.xflip = torchvision.transforms.v2.RandomHorizontalFlip(p=0.5)

    def __call__(self, x):
        if self.mode=='train' and self.apply:
            x = self.resize(x)
            if self.mask:
                x = x*self.masking_array
            if np.random.random()>0.15:
                x=torchvision.transforms.v2.functional.adjust_brightness(x,np.amax((0.1,1+np.random.randn()*0.25)))
            if np.random.random()>0.15:
                x=torchvision.transforms.v2.functional.adjust_contrast(x,np.amax((0.1,1+np.random.randn()*0.25)))
            if np.random.random()>0.15:
                x = self.randomResizedCrop(x)
                x = self.randomRotate(x)
                x=self.xflip(x)
        else:
            x = self.resize(x)
            if self.mask:
                x = x*self.masking_array
        return x.type(torch.float)