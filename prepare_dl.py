import os
import csv
import json
import numpy as np

# Take all the unique (lat, long) in wealth_data.csv
# Randomly shuffle points
# Take 1000 of them
# create a points.txt
# Note: points.txt is longitude / latitude
root = "data"
s = set()
wealth = {}
with open(os.path.join(root, "wealth_data.csv"), 'r') as f:
	reader = csv.reader(f, delimiter=",")
	for i, line in enumerate(reader):
		if i == 0: continue # skip headers
		s.add(( round(float(line[8]), 5), round(float(line[9]), 5) ))
		wealth[(round(float(line[8]), 5), round(float(line[9]), 5))] = float(line[12])

inds = list(range(len(s)))
inds = np.random.choice(inds, 1000, replace=False)

npos = 0
nneg = 0

all_points = []
for i, val in enumerate(s):
	if i in inds:
		all_points.append(list(val))
		if wealth[val] < 0:
			nneg += 1
		else:
			npos += 1

print("Neg", nneg, "Pos", npos, "Total", len(all_points))
with open(os.path.join(root, "points.txt"), 'w') as f:
	json.dump(all_points, f)