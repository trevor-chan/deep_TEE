#%%
import sys
sys.path.append('..')
import re
import os
import random
import numpy as np
from PIL import Image
from lib.dicom_utilities import read_dicom, get_dicom_image_data, save_fig
from lib.subject_utilities import load_excel, aggregate_data
from natsort import natsorted


# parameters
home_path = os.path.expanduser('~/')
data_path = f'../data/first_transfer/8'
data_directories = {'train_path':f'../data/multiview/train','val_path':f'../data/multiview/val','test_path':f'../data/multiview/test'}
random.seed(42)
sample_subset = False
num_frames = 50 # unclear what the best frame selection is - some clips longer than others
final_w = 200

test_subjects = []
val_subjects = []


def run(args):
    
    print(args)

    data_directories = {'train_path':f'../data/multiview/train','val_path':f'../data/multiview/val','test_path':f'../data/multiview/test'}


    
    #create a list of subjects

    for subject_path in natsorted(os.listdir(data_path)):
        if subject_path not in args: continue
        
        for image_path in os.listdir(f'{data_path}/{subject_path}'):
            
            print(f'Processing {data_path}/{subject_path}/{image_path}')
            try:
                data = get_dicom_image_data(read_dicom(f'{data_path}/{subject_path}/{image_path}'))
                
                if len(data.shape) != 4: continue # filter out 2D stills
                                
                if sample_subset:
                    frame_indices = random.sample(range(0, data.shape[0]-1), num_frames)
                else:
                    frame_indices = range(0, data.shape[0]-1)
                
                    
                for frame_index in frame_indices:
                    try:
                        original_img = Image.fromarray(data[frame_index,:,:,:])#.convert('RGB')
                        final_h = int(600 / 800 * final_w)
                        # final_img = Image.alpha_composite(original_img, mask).convert('L').resize([final_w, final_h], Image.BICUBIC) # For conversion to a grayscale image with borders masked
                        # final_img = Image.alpha_composite(original_img, mask).convert('RGB').resize([final_w, final_h], Image.BICUBIC) # For conversion to a color image with borders masked
                        final_img = original_img.resize([final_w, final_h], Image.BICUBIC) # For conversion to a color image without masking

                        savepath = data_directories['test_path'] if subject_path in test_subjects else data_directories['val_path'] if subject_path in val_subjects else data_directories['train_path']
                        os.makedirs(f'{savepath}/{subject_path}/{image_path}', exist_ok = True)
                        final_img.save(f'{savepath}/{subject_path}/{image_path}/{str(frame_index).zfill(4)}.png')

                    except Exception as e:
                        print(f'\033[91mException processing subject {subject_path} dicom {image_path} frame_index {frame_index} \n{e}\033[0m') #Throwing error here
                print(f'Saved subject {subject_path} dicom {image_path}')
            except Exception as e:
                print(f'\033[91mException processing subject {subject_path} dicom {image_path} \n{e}\033[0m')
    print('Done!')