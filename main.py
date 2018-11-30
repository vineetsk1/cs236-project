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
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=1e-5, type=float)

parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--gpu_num', default="0", type=str)
parser.add_argument('--save_model_weights', default=0, type=int)
parser.add_argument('--experiment_name', type=str, default=None,
                    help='Experiment name to save logs and checkpoints under. If not specified, will be using timestamp instead.')




args = parser.parse_args()
print(args.experiment_name)
experiment_name = args.experiment_name
if experiment_name is None:
    now = datetime.datetime.now()
    experiment_name = now.strftime('%m%d_%H%M')

model_path = os.path.join('weights', experiment_name)


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

	print("Initializing train dataset")
	train_dset, train_loader = data_loader(args.dataset_folder, train_dir, args.batch_size)
	print("Initializing val dataset")
	val_dset, val_loader = data_loader(args.dataset_folder, val_dir, args.batch_size)
	print("Training for %d" % args.num_epochs)
	print("Arguments", args.__dict__)

	model = PredictFromNightBaseline() 
	# model = PredictFromDayBaseline()
	# model = PredictBaseline()
	model.type(float_dtype)
	print(model)

	optimizer = optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=args.learning_rate)
	# criterion = nn.BCELoss()
	criterion = nn.CrossEntropyLoss()

	max_val_acc = 0.0

	for epoch in range(args.num_epochs):
		gc.collect()

		# Train epoch
		model.train()
		loss_avg = RunningAverage()
		acc_avg = RunningAverage()
		with tqdm(total=len(train_loader)) as t:
			for i, train_batch in enumerate(train_loader):
				train_batch = [tensor.cuda() if args.use_gpu else tensor for tensor in train_batch]
				X_day, X_night, Y = train_batch

				out = model(X_day, X_night)
				loss = criterion(out, Y)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				acc_avg.update_step(calc_accuracy(out, Y), Y.shape[0])
				loss_avg.update_step(loss.item(), Y.shape[0])
				t.set_postfix(loss='{:05.3f}'.format(loss_avg()), acc='{:05.3f}'.format(acc_avg()))
				t.update()
#why doesnt the code go here
		# Val metrics
		model.eval()
		val_loss = RunningAverage()
		val_acc = RunningAverage()
		for i, val_batch in enumerate(val_loader):
			val_batch = [tensor.cuda() if args.use_gpu else tensor for tensor in val_batch]
			X_day, X_night, Y = val_batch

			out = model(X_day, X_night)
			loss = criterion(out, Y)

			val_loss.update_step(loss.item(), Y.shape[0])
			val_acc.update_step(calc_accuracy(out, Y), Y.shape[0])

		metrics_string = "Loss: {:05.3f} ; Acc: {:05.3f}".format(loss_avg(), acc_avg())
		val_metrics = "Loss: {:05.3f} ; Acc: {:05.3f}".format(val_loss(), val_acc())
		print("Epoch [%d/%d] - Train -" % (epoch+1, args.num_epochs), metrics_string, "- Val -", val_metrics)

		if val_acc() > max_val_acc and args.save_model_weights:
			torch.save(model.state_dict(), os.path.join(model_path, epoch))

if __name__ == '__main__':
	main(args)