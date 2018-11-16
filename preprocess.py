import os
import csv
import sys
import json
import random
import shutil
import numpy as np
from shutil import copyfile
from tqdm import tqdm
from PIL import Image

# Setup directories in data/
root = "data"
dirs = ["resize", "train", "val", "test", "npy"]
for directory in dirs:
	if os.path.exists(os.path.join(root, directory)):
		shutil.rmtree(os.path.join(root, directory))
	os.makedirs(os.path.join(root, directory))

# Resize all the files in raw/ and save
raw_dir = "raw"
width = 256
count = 0

for f in tqdm(os.listdir(os.path.join(root, raw_dir))):
	if not f.endswith(".tif") or not f.startswith("day_"):
		continue
	idx = int(f[f.find("_")+1:f.find("_", f.find("_")+1)])
	
	day_name = f[:f.find("_", f.find("_")+1)] + ".jpg"
	night_name = "night_" + day_name[day_name.find("_")+1:]

	img = Image.open(os.path.join(root, raw_dir, f))
	img = img.resize((width, width), Image.ANTIALIAS)
	img.save(os.path.join(root, dirs[0], day_name))

	img = Image.open(os.path.join(root, raw_dir, "night_" + f[f.find("_")+1:]))
	img = img.resize((width, width), Image.ANTIALIAS)
	img.save(os.path.join(root, dirs[0], night_name))

	count += 1

# Split into train/val and copy
idx = list(range(count))
random.shuffle(idx)
pct_train = 0.8
pct_val = 0.1
for i in range(len(idx)):
	img_id = idx[i]
	if (i+1) <= pct_train * len(idx):
		folder_name = dirs[1]
	elif (i+1) <= (pct_train+pct_val) * len(idx):
		folder_name = dirs[2]
	else:
		folder_name = dirs[3]
	
	copyfile(os.path.join(root, dirs[0], "day_%d.jpg" % img_id), 
		os.path.join(root, folder_name, "day_%d.jpg" % img_id))

	copyfile(os.path.join(root, dirs[0], "night_%d.jpg" % img_id), 
		os.path.join(root, folder_name, "night_%d.jpg" % img_id))
	
# Load label data from points.txt
# and wealth_data.csv and merge them using
# find_wealth
points = []
with open(os.path.join(root, "points.txt"), 'r') as f:
	points = json.load(f)

wealth = {}
with open(os.path.join(root, "wealth_data.csv"), 'r') as f:
	reader = csv.reader(f, delimiter=",")
	for i, line in enumerate(reader):
		if i == 0: continue # skip headers
		wealth[(round(float(line[8]), 5), round(float(line[9]), 5))] = float(line[12]) > 0

def find_wealth(idx):
	lat, lon = points[idx]
	lat, lon = round(lat, 5), round(lon, 5)
	return wealth[lat, lon]

def load_image(path):
	img = Image.open(path)
	img.load()
	data = np.asarray(img)
	return data

# Construct .npy files
# X_day = for each f in train/day_*.jpg
# X_night = for each f in train/night_*.jpg
# Y = lookup index in points.txt to find lat/long, lookup in wealth_data

for directory in dirs:
	if directory == "resize" or directory == "npy":
		continue

	X_day = []
	X_night = []
	Y = []

	for f in os.listdir(os.path.join(root, directory)):
		if not f.endswith(".jpg") or not f.startswith("day_"):
			continue
		day_name = f
		night_name = "night_" + day_name[day_name.find("_")+1:]
		idx = int(day_name[day_name.find("_")+1:day_name.find(".")])

		day_img = load_image(os.path.join(root, directory, day_name))
		night_img = load_image(os.path.join(root, directory, night_name))
		label = find_wealth(idx)

		X_day.append(day_img)
		X_night.append(night_img)
		Y.append(label)

	X_day = np.array(X_day)
	X_night = np.array(X_night)
	Y = np.array(Y, dtype=int).reshape((-1, 1))

	np.save(os.path.join(root, dirs[4], "Xday_" + directory + ".npy"), X_day)
	np.save(os.path.join(root, dirs[4], "Xnight_" + directory + ".npy"), X_night)
	np.save(os.path.join(root, dirs[4], "Y_" + directory + ".npy"), Y)
