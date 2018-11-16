import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

class BackboneLayer(nn.Module):
	def __init__(self):
		super(BackboneLayer, self).__init__()

		# VGG16 is:
		# Part 1: (Features)
		# Conv2D(3, 64, 3/3, 1/1, 1/1), ReLu
		# Conv2D(64, 64, 3/3, 1/1, 1/1), ReLu
		# MaxPool2D(2, 2, 0, 1, False)
		# # 
		# Conv2D(64, 128, 3/3, 1/1, 1/1), ReLu
		# Conv2D(128, 128, 3/3, 1/1, 1/1), ReLU
		# MaxPool2D(2, 2, 0, 1, False)
		# 
		# Conv2D(128, 256, 3/3, 1/1, 1/1), ReLu
		# Conv2D(256, 256, 3/3, 1/1, 1/1), ReLU
		# Conv2D(256, 256, 3/3, 1/1, 1/1), ReLU
		# MaxPool2D(2, 2, 0, 1, False)
		# 
		# Conv2D(256, 512, 3/3, 1/1, 1/1), ReLu
		# Conv2D(512, 512, 3/3, 1/1, 1/1), ReLU
		# Conv2D(512, 512, 3/3, 1/1, 1/1), ReLU
		# MaxPool2D(2, 2, 0, 1, False)
		# 
		# Conv2D(512, 512, 3/3, 1/1, 1/1), ReLU
		# Conv2D(512, 512, 3/3, 1/1, 1/1), ReLU
		# Conv2D(512, 512, 3/3, 1/1, 1/1), ReLU
		# MaxPool2D(2, 2, 0, 1, False)
		# Part 2: (Classifier)
		# Linear(25088, 4096, True), ReLU, Dropout(0.5)
		# Linear(4096, 4096, True), ReLU, Dropout(0.5)
		# Linear(4096, 1000, True) -> classifiying to 1000 objs

		orig_model = models.vgg16(pretrained=True)
		self.features = list(orig_model.children())[0] # Just Features
		for child in self.features.children():
		    for param in child.parameters():
		        param.requires_grad = False # Freeze backbone

	def forward(self, x):
		x = self.features(x)
		return x

class PredictFromDayBaseline(nn.Module):
	def __init__(self):
		super(PredictFromDayBaseline, self).__init__()
		self.backbone = BackboneLayer()
		self.fc1 = nn.Linear(512*8*8, 4096, bias=True)
		self.fc2 = nn.Linear(4096, 4096, bias=True)
		self.fc3 = nn.Linear(4096, 1, bias=True)
		self.sig = nn.Sigmoid()

	# use day, not night
	def forward(self, x, _):
		x = self.backbone(x) # n x 512 x 8 x 8
		x = x.view(x.shape[0], -1) # n x 512*8*8
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.sig(self.fc3(x))
		return x

class PredictFromNightBaseline(nn.Module):
	def __init__(self):
		super(PredictFromNightBaseline, self).__init__()
		self.backbone = BackboneLayer()
		self.fc1 = nn.Linear(512*8*8, 4096, bias=True)
		self.fc2 = nn.Linear(4096, 4096, bias=True)
		self.fc3 = nn.Linear(4096, 1, bias=True)
		self.sig = nn.Sigmoid()

	# use night, not day
	def forward(self, _, x):
		x = self.backbone(x) # n x 512 x 8 x 8
		x = x.view(x.shape[0], -1) # n x 512*8*8
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.sig(self.fc3(x))
		return x
