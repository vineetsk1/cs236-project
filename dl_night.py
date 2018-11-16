import ee
import time
import json
ee.Initialize()

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
dataset = dataset.filter(ee.Filter.date('2017-01-01', '2017-01-31'))
dataset = dataset.first()
dataset = dataset.select('avg_rad')

visParams = {"bands":"avg_rad", "min": 0.0, "max": 60.0}
dataset = dataset.visualize(**visParams)

with open("data/points.txt", 'r') as f:
    points = json.load(f)

print("Loaded", len(points), "points")

def single_job(i):
    geometry = [
        [points[i][0]-1, points[i][1]-1], 
        [points[i][0]+1, points[i][1]+1],
        [points[i][0]-1, points[i][1]+1],
        [points[i][0]+1, points[i][1]-1]
    ]
    name = 'night_' + str(i) + '_' + str(points[i][0]) + '_' + str(points[i][1])
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
