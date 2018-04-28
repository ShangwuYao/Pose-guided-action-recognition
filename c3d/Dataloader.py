import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import numpy as np
from scipy.io import loadmat
from random import shuffle
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from sklearn.model_selection import train_test_split



class JHMDB(torch.utils.data.Dataset):
    def __init__(self, video_pathes, mask_pathes, pose_pathes, class_names):
        
        self.data = {'video': [], 'label': [], 'mask':[], 'pose':[], 'scale':[]}
        
        self.classdict = {}
        for i, x in enumerate(class_names):
            self.classdict[x] = i

        video_num=len(video_pathes)
        mask_num=len(mask_pathes)

        for i in range(video_num):
            video=[]
            cap = cv2.VideoCapture(video_pathes[i])
            has_frame=True
            while(has_frame):
                _, frame = cap.read()
                has_frame = frame is not None

                if has_frame:
                    frame = cv2.resize(frame, (112, 112), interpolation = cv2.INTER_CUBIC) # (112, 112, 3)
                    #frame = np.transpose(frame, (2, 0, 1)) # (3, 112, 112)
                    video.append(frame)
            cap.release()
            self.data['video'].append(video)

            mask_mat = loadmat(mask_pathes[i]) 
            masks = cv2.resize(mask_mat['part_mask'], (112, 112), interpolation = cv2.INTER_CUBIC) # (112, 112, F)
            self.data['mask'].append(masks)
            self.data['label'].append(video_pathes[i].split('/')[-2])
            pose_mat = loadmat(pose_pathes[i])['pos_img']
            scale = loadmat(pose_pathes[i])['scale']
            self.data['pose'].append(pose_mat)
            self.data['scale'].append(scale[0]) # redundant dim
            
            
    def _compute_mean(self):
        meanstd_file = './data/jhmdbmean'
        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for videos in self.data['video']:
                for img in videos:
                    # CxHxW
                    print(img)
                    print(np.reshape(img, (img.shape(0), -1)))
                    mean += np.reshape(img, (img.shape(0), -1)).mean(1)
                    std += np.reshape(img, (img.shape(0), -1)).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
            
        return meanstd['mean'], meanstd['std']
    
    def __getitem__(self, index):
        # video (F, C, 112, 112) to be reshaped on the fly
        # label scala, 
        # mask (112, 112, F)
        # pose (2, 15, F)
        # scale (F)
        # randomly select 15 consecutive frames (F = 15)
        frame_num = self.data['mask'][index].shape[2]
        F = 15
        start_frame = np.random.randint(0, high=frame_num-F+1)
        
        # change pose position according to resize
        pose_data = torch.from_numpy(self.data['pose'][index][:,:,start_frame:start_frame+F].astype('float'))
        pose_data[0,:,:] = pose_data[0,:,:] * 112 / 240
        pose_data[1,:,:] = pose_data[1,:,:] * 112 / 320
        
        return torch.from_numpy(np.array(self.data['video'][index][start_frame:start_frame+F])).int(), \
            torch.LongTensor([self.classdict[self.data['label'][index]]]), \
            torch.from_numpy(self.data['mask'][index][:,:,start_frame:start_frame+F].astype('float')), \
            pose_data, \
            torch.from_numpy(self.data['scale'][index][start_frame:start_frame+F].astype('float'))
        
    def __len__(self):
        return len(self.data['scale'])


video_rootdir="./ReCompress_Videos"
mask_rootdir="./puppet_mask"
pose_rootdir="./joint_positions"

video_pathes=[]
mask_pathes=[]
pose_pathes=[]
for root, dirs, files in os.walk(video_rootdir):
    for file in files:
        if file[0].startswith(".") or root.endswith('.AppleDouble'):
            continue
        video_pathes.append(os.path.join(root, file))

for root, dirs, files in os.walk(mask_rootdir):
    for file in files:
        if file[0].startswith(".") or root.endswith('.AppleDouble'):
            continue
        mask_pathes.append(os.path.join(root, file)) 

for root, dirs, files in os.walk(pose_rootdir):
    for file in files:
        if file[0].startswith(".") or root.endswith('.AppleDouble'):
            continue
        pose_pathes.append(os.path.join(root, file))    

video_pathes_train, video_pathes_valid, mask_pathes_train, mask_pathes_valid, pose_pathes_train, pose_pathes_valid = \
    train_test_split(video_pathes, mask_pathes, pose_pathes, test_size=0.01)
class_names=[name for name in os.listdir(video_rootdir) if not name.startswith(".")]

valid_dataset = JHMDB(video_pathes_valid, mask_pathes_valid, pose_pathes_valid, class_names)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
    batch_size=5, shuffle=True)
    #num_workers=2, pin_memory=True)

