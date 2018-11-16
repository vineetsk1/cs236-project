import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def data_collate(data):

	x1, x2, y = list(zip(*data))
	x1 = torch.cat(x1, dim=0).type(torch.FloatTensor)
	x2 = torch.cat(x2, dim=0).type(torch.FloatTensor)
	y = torch.cat(y, dim=0).type(torch.FloatTensor)

	return (x1, x2, y)

def data_loader(data_dir, data_split, batch_size):
	dset = PovertyDataset(data_dir, data_split)
	loader = DataLoader(
		dset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=4,
		collate_fn=data_collate)
	return dset, loader

class PovertyDataset(Dataset):

	def __init__(self, data_dir, data_split):
		super(PovertyDataset, self).__init__()

		self.X1 = os.path.join(data_dir, "Xday_%s.npy" % data_split)
		self.X2 = os.path.join(data_dir, "Xnight_%s.npy" % data_split)
		self.Y = os.path.join(data_dir, "Y_%s.npy" % data_split)

		self.X1 = np.load(self.X1)
		self.X2 = np.load(self.X2)
		self.X2 = self.X2.reshape((self.X2.shape[0], self.X2.shape[1], self.X2.shape[2], 1))
		self.X2 = np.repeat(self.X2, 3, axis=3)
		self.Y = torch.from_numpy(np.load(self.Y))

		# Image should be (batch, channels, width, height)
		# Also transform with mean/std for VGG net

		self.transform = transforms.Compose([
			transforms.ToTensor(), # Normalize [0, 1]
			transforms.Normalize(mean=[0.485, 0.456, 0.406], # Mean/STD scaling
                                     std=[0.229, 0.224, 0.225])
		])


	def __len__(self):
		return self.Y.shape[0]

	def __getitem__(self, idx):
		return self.transform(self.X1[idx]).float().unsqueeze(0), \
			   self.transform(self.X2[idx]).float().unsqueeze(0), \
			   self.Y[idx:idx+1]
