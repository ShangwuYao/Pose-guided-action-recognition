import numpy as np
import os
import pandas as pd
import torch
import torch.utils.data
import argparse
import time
import csv
import torch.nn as nn
from torch import optim
import math
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset
from collections import namedtuple
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
import torch.nn.functional as F
from torch.utils.data.dataloader import _use_shared_memory
import time
from torch.optim.lr_scheduler import StepLR
import re
from sklearn.model_selection import train_test_split
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import numpy as np
from scipy.io import loadmat
from random import shuffle
from torch.utils.data import Dataset, DataLoader
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.draw




class JHMDB_image(torch.utils.data.Dataset):
    def __init__(self, video_pathes, mask_pathes, pose_pathes, class_names):
        
        self.data = {'image': [], 'label': [], 'mask':[], 'pose':[], 'scale':[]}
        
        self.classdict = {}
        for i, x in enumerate(class_names):
            self.classdict[x] = i

        video_num=len(video_pathes)
        mask_num=len(mask_pathes)

        for i in range(video_num):
            cap = cv2.VideoCapture(video_pathes[i])
            has_frame=True
            frame_cnt = 0
            while(has_frame):
                _, frame = cap.read()
                has_frame = frame is not None

                if has_frame:
                    frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_CUBIC) # (112, 112, 3)
                    #frame = np.transpose(frame, (2, 0, 1)) # (3, 112, 112)
                    self.data['image'].append(frame)
                    frame_cnt += 1
            cap.release()
            
            mask_mat = loadmat(mask_pathes[i]) 
            masks = cv2.resize(mask_mat['part_mask'], (224, 224), interpolation = cv2.INTER_CUBIC) # (112, 112, F)
            
            pose_mat = loadmat(pose_pathes[i])['pos_img']
            scale = loadmat(pose_pathes[i])['scale']
            
            for j in range(frame_cnt):
                self.data['mask'].append(masks[:,:,j])
                self.data['label'].append(video_pathes[i].split('/')[-2])

                self.data['pose'].append(pose_mat[:,:,j])
                self.data['scale'].append(scale[0][j]) # redundant dim
            
            
    def _compute_mean(self):
        meanstd_file = './data/jhmdbmean'
        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            #mean = torch.zeros(3)
            #std = torch.zeros(3)
            mean = np.zeros(3)
            std = np.zeros(3)
            cnt = 0
            for videos in self.data['video']:
                for img in videos:
                    # CxHxW
                    mean += np.reshape(img, (-1, img.shape[-1])).mean(0)
                    std += np.reshape(img, (-1, img.shape[-1])).std(0) 
                    cnt += 1
            mean /= cnt
            std /= cnt
            mean = torch.FloatTensor(mean)
            std = torch.FloatTensor(std)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
            
        print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
        print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
            
        return meanstd['mean'], meanstd['std']
    
    def __getitem__(self, index):
        # image (C, 112, 112) to be reshaped on the fly
        # label scala, 
        # mask (112, 112)
        # pose (2, 15)
        # scale scala
        
        pose_data = torch.from_numpy(self.data['pose'][index].astype('float'))
        
        # change pose position according to resize
        pose_data[0,:] = pose_data[0,:] * 7 / 320
        pose_data[1,:] = pose_data[1,:] * 7 / 240
        
        # features, action_label, mask_label, pose_label, scale
        return torch.from_numpy(np.array(self.data['image'][index])).float(), \
            torch.LongTensor([self.classdict[self.data['label'][index]]]), \
            torch.from_numpy(self.data['mask'][index].astype('float')), \
            pose_data, \
            self.data['scale'][index]
            #torch.FloatTensor([self.data['scale'][index].astype('float')])
        
    def __len__(self):
        return len(self.data['scale'])


class JHMDB_video(torch.utils.data.Dataset):
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
                    frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_CUBIC) # (112, 112, 3)
                    #frame = np.transpose(frame, (2, 0, 1)) # (3, 112, 112)
                    video.append(frame)
            cap.release()
            self.data['video'].append(video)

            mask_mat = loadmat(mask_pathes[i]) 
            masks = cv2.resize(mask_mat['part_mask'], (224, 224), interpolation = cv2.INTER_CUBIC) # (112, 112, F)
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
            #mean = torch.zeros(3)
            #std = torch.zeros(3)
            mean = np.zeros(3)
            std = np.zeros(3)
            cnt = 0
            for videos in self.data['video']:
                for img in videos:
                    # CxHxW
                    mean += np.reshape(img, (-1, img.shape[-1])).mean(0)
                    std += np.reshape(img, (-1, img.shape[-1])).std(0) 
                    cnt += 1
            mean /= cnt
            std /= cnt
            mean = torch.FloatTensor(mean)
            std = torch.FloatTensor(std)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
            
        print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
        print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
            
        return meanstd['mean'], meanstd['std']
    
    def __getitem__(self, index):
        # video (F, 224, 224, C) to be reshaped on the fly
        # label scala, 
        # mask (224, 224, F)
        # pose (2, 15, F)
        # scale (F)
        # 1x40x224x224x3, 1x1, 1x224x224x40, 1x2x15x40
        
        pose_data = torch.from_numpy(self.data['pose'][index].astype('float'))
        
        # change pose position according to resize
        pose_data[0,:,:] = pose_data[0,:,:] * 224 / 320
        pose_data[1,:,:] = pose_data[1,:,:] * 224 / 240
        
        # features, action_label, mask_label, pose_label, scale
        return torch.from_numpy(np.array(self.data['video'][index])).float(), \
            torch.LongTensor([self.classdict[self.data['label'][index]]]), \
            torch.from_numpy(self.data['mask'][index].astype('float')), \
            pose_data, \
            #torch.from_numpy(self.data['scale'][index][selected_frame].astype('float'))
        
    def __len__(self):
        return len(self.data['scale'])


# use some code from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet101']


model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x




def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def to_tensor(numpy_array, datatype):
    # Numpy array -> Tensor
    if datatype == 'int':
        return torch.from_numpy(numpy_array).int()
    elif datatype == 'long':
        return torch.from_numpy(numpy_array).long()
    else:
        return torch.from_numpy(numpy_array).float()


def to_variable(tensor, cpu=False):
    # Tensor -> Variable (on GPU if possible)
    #print(type(tensor))
    if torch.cuda.is_available() and not cpu:
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


class LSTMModel(torch.nn.Module):
    '''
    Paper: Listen, attend and spell
    '''
    def __init__(self, args, dataloader, valid_dataloader):
        super(LSTMModel, self).__init__()
        
        num_pose_keypoints = 15
        NUM_CLASSES = 393
        self.dropout = 0.5
        
        self.ResnetModel = resnet101(True)
        
        self.BU_attention = nn.Conv2d(2048, 1, (1, 1))
        
        #self.pose_pre_logits = nn.Conv2d(2048, 768, (1, 1))
        #self.relu_layer = nn.ReLU()
        #self.pose_logits = nn.Conv2d(768, num_pose_keypoints, (1, 1))
        #self.dropout_layer = nn.Dropout(self.dropout)

        self.TD_attention = nn.Conv2d(2048 + num_pose_keypoints, NUM_CLASSES, (1, 1))
        #self.TD_attention = nn.Conv2d(2048, NUM_CLASSES, (1, 1))

        # initialization
        #self.apply(wsj_initializer)
        
        self.args = args
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion_action_xentropyloss = nn.CrossEntropyLoss()
        self.criterion_pose_l2loss = nn.MSELoss()
        self.best_validation_acc = 0.
        self.model_param_str = 'weights'
        self.optimizer = optim.Adam(self.parameters(), lr=args.init_lr)
        self.k = 10

        if torch.cuda.is_available():
            self.cuda()

    def adjust_lr(self, epoch):
        lr = self.args.init_lr * (0.1 ** (epoch // self.k))
        lr = max(1e-5, lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print("----------adjusting learning rate: {}----------".format(lr))

    def forward(self, input_features, pose_heatmaps):
        start_time = time.time()
        
        # resnet extract feature
        extracted_features = self.ResnetModel(input_features) # (B, 2048, 7, 7)
        
        # bottom up
        BU_logits = self.BU_attention(extracted_features)
        
        # top down
        #pre_pose_logits = self.relu_layer(self.pose_pre_logits(extracted_features))
        #pose_logits = self.pose_logits(pre_pose_logits) # (B, 16, 7, 7)
        
        # append extracted features
        #pose_logits_with_extracted_features = self.dropout_layer(torch.cat((extracted_features, pose_heatmaps), 1))
        pose_logits_with_extracted_features = torch.cat((extracted_features, pose_heatmaps), 1)
        TD_logits = self.TD_attention(pose_logits_with_extracted_features)
        
        #TD_logits = self.TD_attention(extracted_features)
        
        pre_logits = BU_logits * TD_logits
        action_score = torch.mean(pre_logits.view(pre_logits.size(0), pre_logits.size(1), -1), dim=2)

        decoder_time = time.time()
        #print("--- decoder %s seconds ---" % (decoder_time - encoder_time))
        # (B, 393), (B, 16, 7, 7)
        return action_score

    def model_train(self, freeze=True, use_mask=False):
        if freeze:
            print("--------freeze--------")
            for param in self.ResnetModel.parameters():
                param.requires_grad = False
        else:
            print("--------unfreeze--------")
            for param in self.ResnetModel.parameters():
                param.requires_grad = True
        
        for i in range(self.args.start_epoch, self.args.epochs):
            print("---------epoch {}---------".format(i))
            start_time = time.time()
            self.train()

            self.adjust_lr(i)

            losses = 0
            total_cnt = 0
            
            for features, action_label, mask, pose, scale in self.dataloader: # TODO: change back
                fwd_time = time.time()
                self.zero_grad()
        
                # change to frame level prediction
                features = features.contiguous().transpose(3, 2).transpose(2, 1).contiguous()
                #f_size = features.size()
                #features = features.view(f_size[0] * f_size[1], f_size[2], f_size[3], f_size[4])
                features = to_variable(features)
                #print(features.size()) # (B, 3, 224, 224)
                
                # apply hard attention using mask
                if use_mask:
                    # mask (B, 224, 224)
                    mask = mask.unsqueeze(1)
                    mask = mask.expand_as(features)

                    import pdb; pdb.set_trace()
                    visualize_tensor(features[0].data)
                    features = features * to_variable(mask)
                    visualize_tensor(features[0].data)
                    
                # apply hard attention using pose
                pose_heatmaps = []
                # calculate gaussian independently, sum over, clip
                # pose (B, 2, 15)
                #pose = pose.transpose(1, 2).transpose(2, 3).transpose(1, 2).contiguous() # (B, F, 15, 2)
                pose = pose.transpose(1, 2).contiguous() # (B, 15, 2)
                p_size = pose.size()
                #pose = pose.view(p_size[0] * p_size[1], p_size[2], p_size[3]).numpy() # (B x F, 15, 2)

                # should be heatmap of shape (B, 224, 224)
                for scale_idx, pose_15 in enumerate(pose):
                    pose_heatmap = np.zeros((15, 7, 7))
                    for idx, pose_pt in enumerate(pose_15):
                        img = np.zeros((7, 7))
                        #pose_heatmap[i] = gaussian(img, pose_pt, self.args.sigma)
                        pose_heatmap[idx] = circle(img, pose_pt, 1, 1)

                    #pose_heatmap = pose_heatmap.sum(0, keepdims=True).clip(0, 1)
                    pose_heatmaps.append(pose_heatmap)
                pose_heatmaps = to_variable(torch.from_numpy(np.array(pose_heatmaps)).float())
                
                #print(pose_heatmaps)
                
                #F = 15
                #action_label = action_label.repeat(1,F).view(-1)
                action_label = action_label.view(-1)
                action_score = self.forward(features, pose_heatmaps)

                action_xentropyloss = self.criterion_action_xentropyloss(action_score, to_variable(action_label))
                loss = action_xentropyloss
                # TODO: only for valid bit is 1
                #pose_L2loss = self.criterion_pose_l2loss(pose_logits, to_variable(pose_label))
                #loss = action_xentropyloss + pose_L2loss
                
                total_cnt += 1
                losses += loss.data[0]
                #print("epoch {}, loss: {}".format(i, loss.data[0]))

                #bwd_time = time.time()
                #print("--- fwd %s seconds ---" % (bwd_time - fwd_time)) 
                
                loss.backward() 

                self.optimizer.step()
                #print("--- bwd %s seconds ---" % (time.time() - bwd_time))  
            print("training loss: {}".format(losses / total_cnt))
            validation_acc = self.evaluate()
            
            print("--------saving model--------")
            self.model_param_str = \
                'att_img_pose_epoch_{}_loss_{}_valloss_{}'.format(
                    i, losses / total_cnt, validation_acc)
            torch.save(self.state_dict(), self.model_param_str + '.t7')
            if validation_acc > self.best_validation_acc:
                self.best_validation_acc = validation_acc

            print("--- %s seconds ---" % (time.time() - start_time))    

        return self.model_param_str     
    def evaluate(self):
        self.eval()

        #losses = 0
        total_cnt = 0
        true = 0
        false = 0
        for features, action_label, _, pose in self.valid_dataloader:
            # 1x40x224x224x3, 1x1, 1x224x224x40, 1x2x15x40
            features = features[0]
            pose = pose[0]
            
            # change to frame level prediction
            #features = features.contiguous().transpose(4, 3).transpose(3, 2).contiguous()
            #f_size = features.size()
            #features = features.view(f_size[0] * f_size[1], f_size[2], f_size[3], f_size[4])
            ##print(features.size()) # (B * F, 3, 224, 224)
            features = features.contiguous().transpose(3, 2).transpose(2, 1).contiguous()
            features = to_variable(features)
            
            # apply hard attention using pose
            pose_heatmaps = []
            # calculate gaussian independently, sum over, clip
            # pose (2, 15, 40)
            #pose = pose.transpose(1, 2).transpose(2, 3).transpose(1, 2).contiguous() # (B, F, 15, 2)
            #print("pose", pose.size())
            pose = pose.transpose(1, 2).transpose(0, 1).transpose(1, 2).contiguous() # (B, 15, 2)
            p_size = pose.size()
            #pose = pose.view(p_size[0] * p_size[1], p_size[2], p_size[3]).numpy() # (B x F, 15, 2)

            #print("p_size", p_size) # 40, 15, 2
            # should be heatmap of shape (B, 224, 224)
            for scale_idx, pose_15 in enumerate(pose):
                pose_heatmap = np.zeros((15, 7, 7))
                for idx, pose_pt in enumerate(pose_15):
                    img = np.zeros((7, 7))
                    #pose_heatmap[i] = gaussian(img, pose_pt, self.args.sigma)
                    pose_heatmap[idx] = circle(img, pose_pt, 1, 1)

                #pose_heatmap = pose_heatmap.sum(0, keepdims=True).clip(0, 1)
                pose_heatmaps.append(pose_heatmap)
            pose_heatmaps = to_variable(torch.from_numpy(np.array(pose_heatmaps)).float())

            #F = 15
            #action_label = action_label.repeat(1,F).view(-1)
            action_label = action_label.view(-1)
            action_score = self.forward(features, pose_heatmaps)

            #action_xentropyloss = self.criterion_action_xentropyloss(action_score, to_variable(action_label))
            #loss = action_xentropyloss

            #total_cnt += 1
            #losses += loss.data[0]
            #print("validation loss: {}".format(loss.data[0]))
            
            #print("action_score", action_score.size())
            action = torch.mean(action_score, dim=0, keepdim=True)
            action = torch.max(action, dim=1)[1]
            action = action.data
            
            #print("action: ", action, action_label)
            if action.data.cpu()[0] == action_label[0]:
                true += 1
            else:
                false += 1

        #losses /= total_cnt
        #print("validation loss: {}".format(losses))
        acc = true / (true + false)
        print("validation accuracy: {}".format(acc))
        return acc 

def visualize_tensor(x):
    if x.size(0) == 3:
        x = x.transpose(0, 1).transpose(1, 2)
    imgplot = plt.imshow(x.numpy())


def circle(img, pt, color, radius):
    # Draw a circle
    # Mostly a convenient wrapper for skimage.draw.circle

    rr, cc = skimage.draw.circle(pt[1], pt[0], radius, img.shape)
    img[rr, cc] = color
    return img



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

video_pathes = sorted(video_pathes)
mask_pathes = sorted(mask_pathes)
pose_pathes = sorted(pose_pathes)

video_list = [re.findall(r"[\w_]+\d+", x)[0] for x in video_pathes]
mask_list = [re.findall(r"[\w_]+\d+", x)[0] for x in mask_pathes]
pose_list = [re.findall(r"[\w_]+\d+", x)[0] for x in pose_pathes]

# make sure every thing matches
for x, y, z in zip(video_list, mask_list, pose_list):
    assert (x == y) and (x == z)

video_pathes_train, video_pathes_valid, mask_pathes_train, mask_pathes_valid, pose_pathes_train, pose_pathes_valid = \
    train_test_split(video_pathes, mask_pathes, pose_pathes, test_size=0.15, random_state=42)
class_names=[name for name in os.listdir(video_rootdir) if not name.startswith(".")]


train_dataset = JHMDB_image(video_pathes_train, mask_pathes_train, pose_pathes_train, class_names)

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=10, shuffle=True)
    #num_workers=2, pin_memory=True)

valid_dataset = JHMDB_video(video_pathes_valid, mask_pathes_valid, pose_pathes_valid, class_names)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
    batch_size=1, shuffle=True) # can't change to anything other than 1
    #num_workers=2, pin_memory=True)


args = namedtuple('args',
                      [
                          'batch_size',
                          'save_directory',
                          'epochs',
                          'start_epoch',
                          'init_lr',
                          'sigma',
                          'radius',
                          'cuda'])(
        5,
        'output/',
        40,
        0,
        1e-4,
        7,
        7,
        False)

model = LSTMModel(args, train_loader, valid_loader)

#print("--evaluating")
#model.evaluate()
print("--training")
model.model_train(freeze=True, use_mask=False)

#model_param_str = 'att_img_nopose_epoch_8_loss_0.0890108197927475_valloss_0.7626314704062693'
#model.load_state_dict(torch.load(model_param_str + ".t7"))
#model.load_state_dict(torch.load(model_param_str + ".t7", map_location=lambda storage, loc: storage))
model.model_train(freeze=False, use_mask=False)



