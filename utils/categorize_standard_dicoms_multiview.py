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
import glob


# parameters
home_path = os.path.expanduser('~/')
images_path = f'/data/deep_tee/raw_data/train_test/*/*'
data_directories = {'train_path':f'../data/semi-processed/supervised/multiview/large_train',
                    'val_path':f'../data/semi-processed/supervised/multiview/large_val',
                    'test_path':f'../data/semi-processed/supervised/multiview/large_test',
                    'ext_test_path':f'../data/semi-processed/supervised/multiview/large_ext_test'}
in_file = '/data/deep_tee/raw_data/data_collection_dicom_subject_list_with_validation_S1-S51_R1-R1-_2023_09_19.xlsx'
random.seed(42)
sample_subset = False
num_frames = 50 # unclear what the best frame selection is - some clips longer than others
final_w = 400
test_subjects = ['S02', 'S07', 'S15', 'S19', 'S22', 'S27', 'S31', 'S35', 'S41', 'S49',]
ext_test_subjects = ['R1',  'R2',  'R3',  'R4',  'R5',  'R6',  'R7',  'R8',  'R9',  'R10',  'S03',  'S39',]
val_subjects = ['S03', 'S18', 'S29', 'S34', 'S43',]



def run(args):
    
    print(args)
    
    #%% Load the mask
    mask = Image.open('../assets/dicom-mask.png')
    mask.load()

    #%% load subject data
    subject_data = aggregate_data(load_excel(in_file))

    # To be used if filtering out alternatives, necessary to keep them in generally
    # subject_data = subject_data[(subject_data['standard_v_alt'] == 'standard') | (subject_data['standard_v_alt'] == 'standard_color')].copy()
    
    #get a list of all views and create the directories
    views = list(subject_data['label_TEE'].dropna().unique())
        
    # prune rare views and add other label
    
    # All the views
    removals = ['3D_data_lv',
                '3D_data_av',
                '3D_data_av_col',
                '3D_data_mv',
                '3D_data_mv_col',
                '3D_data_pv',
                '3D_data_tv',
                'deep_pv',
                'deep_pv_col',
                'deep_tg',
                'deep_tg_col_a',
                'dta_sax',
                'dta_sax_col',
                'dta_sax_arch',
                'dta_sax_arch_col',
                'dta_sax_lax_xpl',
                'dta_lax_sax_xpl',
                'hep_ivc',
                'hep_ivc_col',
                'ivc_lax_sax_xpl',
                'me_2_ch',
                'me_2_ch_col_m',
                'me_2_lax_xpl',
                'me_4_2_ch_xpl',
                'me_4_2_ch_z_mv_xpl',
                'me_4_ch',
                'me_4_ch_col_m',
                'me_4_ch_col_t',
                'me_4_ch_full_3D',
                'me_4_ch_z_mv',
                'me_4_ch_z_mv_col',
                'me_4_ch_z_tv_col',
                'me_5_ch',
                'me_5_ch_col_m',
                'me_5_ch_col_t',
                'me_av_sax',
                'me_av_sax_col',
                'me_av_z_sax',
                'me_av_z_sax_col',
                'me_bicaval',
                'me_bicaval_col',
                'me_bicaval_mod',
                'me_bicaval_mod_col_t',
                'me_in_out',
                'me_in_out_col_p',
                'me_in_out_col_t',
                'me_laa',
                'me_laa_col',
                'me_laa_xpl',
                'me_laa_xpl_col',
                'me_lax',
                'me_lax_col_a',
                'me_lax_col_m',
                'me_lax_col_mv_av',
                'me_lax_col_z_av',
                'me_lax_col_z_av_xpl',
                'me_lax_col_z_mv',
                'me_lax_z_av',
                'me_lax_z_av_xpl',
                'me_lax_z_mv',
                'me_lupv',
                'me_lupv_col',
                'me_mit_com',
                'me_mit_com_z',
                'me_mit_com_col_m',
                'me_mit_com_col_m_z',
                'me_rupv',
                'me_rupv_col',
                'nan',
                'tg_2_ch',
                'tg_apical_sax',
                'tg_basal_sax',
                'tg_basal_sax_lax_xpl',
                'tg_basal_sax_mv_col',
                'tg_basal_sax_tv',
                'tg_basal_sax_tv_col',
                'tg_basal_lax_tv',
                'tg_basal_lax_tv_col',
                'tg_lax',
                'tg_lax_col_a',
                'tg_mid_pap_lv_sax_2_ch_xpl',
                'tg_mid_pap_sax',
                'ue_asc_ao_lax',
                'ue_asc_ao_sax',
                'ue_asc_ao_sax_lax_xpl',
                'ue_rvot_pv',
                'ue_rvot_pv_col',]

    keeps =    ['deep_tg_col_a',
                'dta_sax',
                'dta_sax_arch',
                'me_2_ch',
                'me_2_ch_col_m',
                'me_4_ch',
                'me_4_ch_col_m',
                'me_4_ch_col_t',
                'me_5_ch',
                'me_5_ch_col_m',
                'me_5_ch_col_t',
                'me_av_sax',
                'me_av_z_sax',
                'me_av_sax_col',
                'me_av_z_sax_col',
                'me_bicaval',
                'me_bicaval_col',
                'me_in_out',
                'me_in_out_col_p',
                'me_in_out_col_t',
                'me_lax',
                'me_lax_col_a',
                'me_lax_col_z_av',
                'me_lax_col_m',
                'me_lax_col_z_mv',
                'me_lax_col_mv_av',
                'me_mit_com',
                'me_mit_com_col_m',
                'tg_basal_sax',
                'tg_mid_pap_sax',
                'ue_asc_ao_sax',]
    
    # keep just a subset
    removals = list(set(removals) - set(keeps))

    
    for rem in removals:
        if rem in views: 
            views.remove(rem)
    views.append('other')

    final_list=['deep_tg_col_a',
                'dta_sax',
                'dta_sax_arch',
                'me_2_ch',
                'me_2_ch_col_m',
                'me_4_ch',
                'me_4_ch_col_m',
                'me_4_ch_col_t',
                'me_5_ch',
                'me_5_ch_col',
                'me_av_sax',
                'me_av_sax_col',
                'me_bicaval',
                'me_bicaval_col',
                'me_in_out',
                'me_in_out_col_p',
                'me_in_out_col_t',
                'me_lax',
                'me_lax_col_a',
                'me_lax_col_m',
                'me_lax_col_mv_av',
                'me_mit_com',
                'me_mit_com_col_m',
                'tg_basal_sax',
                'tg_mid_pap_sax',
                'ue_asc_ao_sax',
                'other',]
    
    for path_mode in data_directories.keys():
        path = data_directories[path_mode]
        if not os.path.exists(f'{path}'):
            os.mkdir(f'{path}')
        for view in final_list:
            if not os.path.exists(f'{path}/{view}'):
                os.mkdir(f'{path}/{view}')
    #create a list of subjects
    subjects = list(subject_data['Subject'].unique())
    
    print(subjects)

    for subject_path in natsorted(glob.glob(images_path)):
        if subject_path not in args: 
            continue
        
        print(subject_path)
        subject = os.path.basename(os.path.splitext(subject_path)[0])
        print(subject)
        
        # res = re.search('(^S[0-9]+)', subject_path)
        # if not res: continue
        # subject = res[0]
        if subject not in subjects: continue
        
        subject_image_nums = list(subject_data[subject_data['Subject'] == subject]['Image_num'])
                
        for i,subject_image_path in enumerate(natsorted(glob.glob(f'{subject_path}/*'))):
            print(subject_image_path)
            
            subject_image_num = i+1

            if subject_image_num not in subject_image_nums: continue
            
            print(f'Attempting to process subject {subject} dicom {subject_image_num}')
            try:
                data = get_dicom_image_data(read_dicom(f'{subject_image_path}'))
                                
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

                        savepath = data_directories['test_path'] if subject in test_subjects else data_directories['val_path'] if subject in val_subjects else data_directories['ext_test_path'] if subject in ext_test_subjects else data_directories['train_path']
                        viewname = subject_data.loc[(subject_data['Subject'] == subject) & (subject_data['Image_num'] == subject_image_num)].values[0,1]

                        # Additional view rules:
                        if viewname == 'me_5_ch_col_m': viewname = 'me_5_ch_col'
                        if viewname == 'me_5_ch_col_t': viewname = 'me_5_ch_col'
                        if viewname == 'me_av_z_sax': viewname = 'me_av_sax'
                        if viewname == 'me_av_z_sax_col': viewname = 'me_av_sax_col'
                        if viewname == 'me_rupv_col': viewname = 'other'
                        if viewname == 'me_lax_col_z_mv': viewname ='me_lax_col_m'
                        if viewname == 'me_lax_col_z_av': viewname ='me_lax_col_a'

                        if viewname not in final_list: viewname = 'other'

                        im_path = f"{savepath}/{viewname}"
                                                
                        final_img.save(f'{im_path}/{subject}_{subject_image_num}_{frame_index}.png')
                        # save_fig(f'{im_path}/{subject}_{subject_image_num}_{frame_index}', np.asarray(final_img) )
                    except Exception as e:
                        print(f'\033[91mException processing subject {subject} dicom {subject_image_num} frame_index {frame_index} \n{e}\033[0m') #Throwing error here
                print(f'Saved subject {subject} dicom {subject_image_num} frames {frame_indices}')
            except Exception as e:
                print(f'\033[91mException processing subject {subject} dicom {subject_image_num} \n{e}\033[0m')
    print('Done!')
