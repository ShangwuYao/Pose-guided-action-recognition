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



model_urls = {
	'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

# use some code from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet101']


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

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x




def resnet101(pretrained=False, **kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	if pretrained:
		#model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
		pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
		model_dict = model.state_dict()
		
		# TODO
		# 1. filter out unnecessary keys
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc')}
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict) 
		# 3. load the new state dict
		model.load_state_dict(model_dict)
	return model

# used code from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py


class VGG(nn.Module):

	def __init__(self, features, num_classes=1000, init_weights=True):
		super(VGG, self).__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, num_classes),
		)
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

class AlexNet(nn.Module):

	def __init__(self, num_classes=1000):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(45, 64, kernel_size=11, stride=4, padding=2), # 3 -> 45
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		return x


def alexnet(pretrained=False, **kwargs):
	r"""AlexNet model architecture from the
	`"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = AlexNet(**kwargs)
	print(model)
	if pretrained:
		pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
		model_dict = model.state_dict()
		weights = torch.cat([pretrained_dict['features.0.weight']] * 15, 1).data

		# TODO
		# 1. filter out unnecessary keys
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and
						   not k.startswith('classifier.6') and not k.startswith('features.0.weight')} #or not k.startswith('features.0.weight'))

		pretrained_dict['features.0.weight'] = weights # TODO: not load this layer as well

		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict) 
		# 3. load the new state dict
		model.load_state_dict(model_dict)
	return model


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 45 # TODO
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg16(pretrained=False, **kwargs):
	"""VGG 16-layer model (configuration "D")
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers(cfg['D']), **kwargs)
	if pretrained:
		#model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
		
		pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
		model_dict = model.state_dict()
		weights = torch.cat([pretrained_dict['features.0.weight']] * 15, 1).data
		
		# TODO
		# 1. filter out unnecessary keys
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and
						   not k.startswith('classifier.6')} #or not k.startswith('features.0.weight'))
		
		pretrained_dict['features.0.weight'] = weights
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict) 
		# 3. load the new state dict
		model.load_state_dict(model_dict)
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

class JHMDB_image_video(torch.utils.data.Dataset):
	def __init__(self, video_pathes, mask_pathes, pose_pathes, class_names):
		
		self.data = {'video': [], 'image': [], 'label': [], 'mask':[], 'pose':[], 'scale':[]}
		
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
			self.data['image'].append(video[int(len(video) / 2)])

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
		# video (F, C, 112, 112) to be reshaped on the fly
		# label scala, 
		# mask (112, 112, F)
		# pose (2, 15, F)
		# scale (F)
		
		frame_num = self.data['mask'][index].shape[2]
		F = 15
		# randomly select 15 consecutive frames (F = 15)
		#start_frame = np.random.randint(0, high=frame_num-F+1)
		#pose_data = torch.from_numpy(self.data['pose'][index][:,:,start_frame:start_frame+F].astype('float'))
		
		# OR randomly select 15 frames
		frame_shuffle=list(range(frame_num))
		shuffle(frame_shuffle)
		selected_frame=sorted(frame_shuffle[0:F])
		pose_data = torch.from_numpy(self.data['pose'][index][:,:,selected_frame].astype('float'))
		
		
		# change pose position according to resize
		pose_data[0,:,:] = pose_data[0,:,:] * 224 / 320
		pose_data[1,:,:] = pose_data[1,:,:] * 224 / 240
		
		# features, action_label, mask_label, pose_label, scale
		return torch.from_numpy(self.data['image'][index]).float(), \
			torch.from_numpy(np.array(self.data['video'][index])[selected_frame]).float(), \
			torch.LongTensor([self.classdict[self.data['label'][index]]]), \
			torch.from_numpy(self.data['mask'][index][:,:,selected_frame].astype('float')), \
			pose_data, \
			#torch.from_numpy(self.data['scale'][index][selected_frame].astype('float'))
		
	def __len__(self):
		return len(self.data['scale'])


class TwoStreamModel(torch.nn.Module):
	'''
	Paper: Listen, attend and spell
	'''
	def __init__(self, args, dataloader, valid_dataloader):
		super(TwoStreamModel, self).__init__()
		
		num_pose_keypoints = 16
		NUM_CLASSES = 21
		self.dropout = 0.5
		
		resnet_params = {
			'nb_epochs': args.nb_epochs,
			'batch_size': args.batch_size,
			'learning_rate': args.init_lr,
			'train_dir': args.save_directory,
			'filename': args.filename
		}
		self.Spatial = resnet101(True, num_classes=NUM_CLASSES)
		self.softmax1 = torch.nn.Softmax(-1)
		
		self.Temporal = vgg16(True, num_classes=NUM_CLASSES)
		#self.Temporal = alexnet(True, num_classes=NUM_CLASSES)
		self.softmax2 = torch.nn.Softmax(-1)
		
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
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

		print("----------adjusting learning rate: {}----------".format(lr))

	def forward(self, image_input, video_input):
		start_time = time.time()
		
		#spatial_logits = self.softmax1(self.Spatial(image_input))
		spatial_logits = self.Spatial(image_input)
		
		#temporal_logits = self.softmax2(self.Temporal(video_input))
		temporal_logits = self.Temporal(video_input)
		
		fusion = (spatial_logits + temporal_logits) / 2

		decoder_time = time.time()
		#print("--- decoder %s seconds ---" % (decoder_time - encoder_time))
		return fusion

	def model_train(self, freeze=True, use_mask=False, use_pose=False):
		if freeze:
			# RESNET
			i = 0
			for child in self.Spatial.children():
				if i == 9:
					print("only train:", child)
				else:
					child.requires_grad = False
				i += 1
			
			# VGG16
			#i = 0
			#for child in self.Temporal.children():
			#	j = 0
			#	for grandchild in child.children():
			#		if i == 1 and j == 6:
			#			print("only train:", grandchild)
			#		else:
			#			grandchild.requires_grad = False
			#		j += 1
			#	i += 1
		else:
			for param in self.Spatial.parameters():
				param.requires_grad = True
			for param in self.Temporal.parameters():
				param.requires_grad = True
		
		for i in range(self.args.start_epoch, self.args.nb_epochs):
			print("---------epoch {}---------".format(i))
			start_time = time.time()
			self.train()

			self.adjust_lr(i)

			losses = 0
			total_cnt = 0
			
			for image, video, action_label, mask, pose in self.dataloader: # TODO: change back
				fwd_time = time.time()
				self.zero_grad()
				
				# change to frame level prediction
				video = video.contiguous().transpose(4, 3).transpose(3, 2).contiguous()
				f_size = video.size()
				video = video.view(f_size[0], f_size[1] * f_size[2], f_size[3], f_size[4])
				video = to_variable(video)
				#print(video.size()) # (B, F * 3, 224, 224)
				
				# apply hard attention using mask
				if use_mask:
					# mask (B, 224, 224, F)
					mask = mask.transpose(3, 2).transpose(2, 1).contiguous().float()
					m_size = mask.size()
					mask = mask.repeat(1, 1, 3, 1).view(m_size[0], m_size[1] * 3, m_size[2], m_size[3])
					# mask should be of size(B, F * 3, 224, 224)

					#import pdb; pdb.set_trace()
					#visualize_tensor(video[1, 9:12].data)
					video = video * to_variable(mask) * 9. / 10. + video / 10.
					#visualize_tensor(video[1, 9:12].data)
					
				
				# apply hard attention using pose
				if use_pose:
					# calculate gaussian independently, sum over, clip
					# pose (B, 2, 15, F)
					pose = pose.transpose(1, 2).transpose(2, 3).transpose(1, 2).contiguous() # (B, F, 15, 2)
					p_size = pose.size()
					#pose = pose.view(p_size[0] * p_size[1], p_size[2], p_size[3]).numpy() # (B x F, 15, 2)
					
					# should be heatmap of shape (B, F * 3, 224, 224)
					pose_heatmaps = []
					for pose_15 in pose:
						pose_heatmap = np.zeros((15, 224, 224))
						for i, pose_pt in enumerate(pose_15):
							img = np.zeros((224, 224))
							pose_heatmap[i] = circle(img, pose_pt, self.args.sigma)

						pose_heatmap = pose_heatmap.sum(0, keepdims=True).clip(0, 1)
						pose_heatmaps.append(pose_heatmap)
					pose_heatmaps = torch.from_numpy(np.concatenate(pose_heatmaps, 0)).float()
					print(pose_heatmaps)
					
					visualize_tensor(features[0].data)
					
					import pdb; pdb.set_trace()
					
					pose_heatmaps = pose_heatmaps.unsqueeze(1).expand_as(features)
					features = features * to_variable(pose_heatmaps)
					
					visualize_tensor(features[0].data)
				
				F = 15
				#action_label = action_label.repeat(1,F).view(-1)
				action_label = action_label.view(-1)
				#print("action_label", action_label)

				image = image.transpose(3, 2).transpose(2, 1).contiguous()
				image = to_variable(image)
				#print(image.size())
				action_score = self.forward(image, video)

				action_xentropyloss = self.criterion_action_xentropyloss(action_score, to_variable(action_label))
				loss = action_xentropyloss
				# TODO: only for valid bit is 1
				#pose_L2loss = self.criterion_pose_l2loss(pose_logits, to_variable(pose_label))
				#loss = action_xentropyloss + pose_L2loss
				
				total_cnt += 1
				losses += loss.data[0]
				print("epoch {}, loss: {}".format(i, loss.data[0]))

				#bwd_time = time.time()
				#print("--- fwd %s seconds ---" % (bwd_time - fwd_time)) 
				
				loss.backward() 

				self.optimizer.step()
				#print("--- bwd %s seconds ---" % (time.time() - bwd_time))  
			print("training loss: {}".format(losses / total_cnt))
			validation_acc = self.evaluate(use_mask)
			
			print("--------saving model--------")
			self.model_param_str = \
				'{}_twostream_epoch_{}_loss_{}_valacc_{}'.format(
					self.args.prefix, i, losses / total_cnt, validation_acc)
			torch.save(self.state_dict(), self.model_param_str + '.t7')
			if validation_acc > self.best_validation_acc:
				self.best_validation_acc = validation_acc

			print("--- %s seconds ---" % (time.time() - start_time))    

		return self.model_param_str     
	def evaluate(self, use_mask=False):
		self.eval()

		losses = 0
		total_cnt = 0
		true = 0
		false = 0
		for image, video, action_label, mask, pose in self.valid_dataloader:
			# change to frame level prediction
			video = video.contiguous().transpose(4, 3).transpose(3, 2).contiguous()
			f_size = video.size()
			video = video.view(f_size[0], f_size[1] * f_size[2], f_size[3], f_size[4])
			video = to_variable(video)
			#print(video.size()) # (B, F * 3, 224, 224)
			
			# apply hard attention using mask
			if use_mask:
				# mask (B, 224, 224, F)
				mask = mask.transpose(3, 2).transpose(2, 1).contiguous().float()
				m_size = mask.size()
				mask = mask.repeat(1, 1, 3, 1).view(m_size[0], m_size[1] * 3, m_size[2], m_size[3])
				# mask should be of size(B, F * 3, 224, 224)

				#import pdb; pdb.set_trace()
				#visualize_tensor(video[1, 9:12].data)
				video = video * to_variable(mask) * 3 / 4 + video / 4
				#visualize_tensor(video[1, 9:12].data)

			F = 15
			#action_label = action_label.repeat(1,F).view(-1)
			action_label = action_label.view(-1)
			#print("action_label", action_label)

			image = image.transpose(3, 2).transpose(2, 1).contiguous()
			image = to_variable(image)
			#print(image.size())
			action_score = self.forward(image, video)

			action_xentropyloss = self.criterion_action_xentropyloss(action_score, to_variable(action_label))
			loss = action_xentropyloss

			total_cnt += 1
			losses += loss.data[0]
			#print("validation loss: {}".format(loss.data[0]))
			
			#print("action_score", action_score)
			#action = torch.mean(action_score, dim=1, keepdim=True)
			action = torch.max(action_score, dim=1)[1]
			action = action
			#print("action", action)
			#print("action_label", action_label)
			
			#print("action: ", action, action_label)
			for a1, a2 in zip(action.data.cpu().numpy(), action_label.data.cpu().numpy()):
				if a1 == a2:
					true += 1
				else:
					false += 1

		losses /= total_cnt
		print("validation loss: {}".format(losses))
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
	

def main():

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
		train_test_split(video_pathes, mask_pathes, pose_pathes, test_size=0.2, random_state=42)
	class_names=[name for name in os.listdir(video_rootdir) if not name.startswith(".")]

	train_dataset = JHMDB_image_video(video_pathes_train, mask_pathes_train, pose_pathes_train, class_names)

	train_loader = torch.utils.data.DataLoader(train_dataset,
		batch_size=2, shuffle=True)

	valid_dataset = JHMDB_image_video(video_pathes_valid, mask_pathes_valid, pose_pathes_valid, class_names)

	valid_loader = torch.utils.data.DataLoader(valid_dataset,
		batch_size=16, shuffle=False)
		#num_workers=2, pin_memory=True)

	args = namedtuple('args',
						  [
							  'batch_size',
							  'save_directory',
							  'filename',
							  'prefix',
							  'nb_epochs',
							  'start_epoch',
							  'init_lr',
							  'sigma',
							  'cuda'])(
			2,
			'output/',
			'resnet',
			'notfreeze_loadfirst_nomask_bs_10',
			30,
			0,
			1e-4,
			7,
			False)

	model = TwoStreamModel(args, train_loader, valid_loader)
	#print(model)

	#model_param_str = 'notfreeze_loadfirst_mask_twostream_epoch_15_loss_0.14077964425086975_valacc_0.7043010752688172'
	#model.load_state_dict(torch.load(model_param_str + ".t7"))

	model.model_train(freeze=False, use_mask=True)

if __name__ == '__main__':
	main()