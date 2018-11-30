import argparse
import gc
import os
import sys
import time
import datetime

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from loader import data_loader
from baselines import PredictFromDayBaseline, PredictFromNightBaseline, PredictBaseline
from utils import RunningAverage
import numpy as np

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_folder', default=os.path.join('data', 'npy'), type=str)
parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

args = parser.parse_args()

def get_dtypes(args):
	long_dtype = torch.LongTensor
	float_dtype = torch.FloatTensor
	if args.use_gpu == 1:
		long_dtype = torch.cuda.LongTensor
		float_dtype = torch.cuda.FloatTensor
	return long_dtype, float_dtype

# Sigmoid Accuracy
# def calc_accuracy(outputs, labels):
	# outputs[:, 0] = (outputs[:, 0] >= 0.5).float()
	# return torch.mean((outputs == labels).float()).item()

# Multi Class Accuracy
def calc_accuracy(outputs, labels):
	_, max_outs = outputs.max(1)
	# max_outs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
	# max_labels = np.argmax(labels, axis=1)
	# return np.mean(max_outs == labels)
	return torch.mean((max_outs == labels).float()).item()

def main(args):
	if args.use_gpu == 1:
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
		print("Using GPU", args.gpu_num)
	else:
		print("Not using GPU")

	train_dir = "train"
	val_dir = "val"
	long_dtype, float_dtype = get_dtypes(args)

	print("Initializing val dataset")
	val_dset, val_loader = data_loader(args.dataset_folder, val_dir, args.batch_size)
	print("Arguments", args.__dict__)

	model = PredictFromNightBaseline() 
	# model = PredictFromDayBaseline()
	# model = PredictBaseline()
	model.type(float_dtype)
	model.load_state_dict(torch.load("weights/1130_0100/1"))
	print(model)

	true_labels = []
	pred_labels = []

	my_map = {}
	my_map[0] = {}
	my_map[1] = {}
	my_map[2] = {}
	total_zeros = 0
	total_ones = 0
	total_twos = 0
	all_total = 0

	for i, val_batch in enumerate(val_loader):
		val_batch = [tensor.cuda() if args.use_gpu else tensor for tensor in val_batch]
		X_day, X_night, Y = val_batch

		out = model(X_day, X_night)

		for j in range(Y.shape[0]):
			true_lbl = Y[j]
			_, pred_lbl = out[j].max(1)
			my_map[true_lbl][pred_lbl] = my_map[true_lbl].get(pred_lbl, 0) + 1
			if true_lbl == 0: total_zeros += 1
			if true_lbl == 1: total_ones += 1
			if true_lbl == 2: total_twos += 1
			all_total += 1

	print(total_zeros, total_ones, total_twos, all_total)

	print(my_map)

	my_map[0][0] /= total_zeros
	my_map[0][1] /= total_zeros
	my_map[0][2] /= total_zeros

	my_map[1][0] /= total_ones
	my_map[1][1] /= total_ones
	my_map[1][2] /= total_ones

	my_map[2][0] /= total_twos
	my_map[2][1] /= total_twos
	my_map[2][2] /= total_twos

	print(my_map)

if __name__ == '__main__':
	main(args)
