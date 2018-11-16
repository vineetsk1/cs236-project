import ee
import time
import json
ee.Initialize()

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def maskL8sr(image):
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    qa = image.select('pixel_qa')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask)

dataset = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate('2017-01-01', '2017-12-31').map(maskL8sr).median()
dataset = dataset.select(["B4", "B3", "B2"])

with open("data/points.txt", 'r') as f:
    points = json.load(f)

visParams = {"min": 0, "max": 3000, "gamma": 1.4, "bands": ["B4", "B3", "B2"]}
dataset = dataset.visualize(**visParams)

print("Loaded", len(points), "points")

def single_job(i):
    geometry = [
        [points[i][0]-1, points[i][1]-1], 
        [points[i][0]+1, points[i][1]+1],
        [points[i][0]-1, points[i][1]+1],
        [points[i][0]+1, points[i][1]-1]
    ]
    name = 'day_' + str(i) + '_' + str(points[i][0]) + '_' + str(points[i][1])
    name = name.replace(".", "_")
    task = ee.batch.Export.image.toDrive(dataset, description=name, scale=250, region=geometry, folder='geoDataF', maxPixels=1e9)
    task.start()
    while task.active():
        time.sleep(5)
    print("Job", i, "completed")
    return "Done"

pool = ThreadPoolExecutor(10)

futures = []
for i in range(len(points)):
    future = pool.submit(single_job, (i))
    futures.append(future)

concurrent.futures.wait(futures)
print("All jobs completed.")
