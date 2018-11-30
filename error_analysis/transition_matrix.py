import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from utils import activities
from visualization import heatmap, annotate_heatmap



output_image_path = 'confusion_matrix.png'


def get_time_from_path(image_path):
  local_path = image_path.split('/')[1]
  timestamp = int(local_path[:-4]) # Remove the ending '.png'
  timestamp /= 1000
  real_time = datetime.utcfromtimestamp(timestamp - 6 * 60 * 60) # Utah is 6 hours ahead of UTC
  return real_time


def read_data(path):
  data = []
  with open(path, 'r') as data_file:
    header = next(data_file)
    for line in data_file:
      _, _, image_path, label = line.strip().split(' ')
      image_time = get_time_from_path(image_path)
      label = int(label)
      data.append((image_time, label))
  
  return data


def calculate_transitions_matrix(data):
  excluded_activities = ['negative', 'delete']
  num_states = len(activities) - len(excluded_activities)
  transitions_matrix = np.zeros((num_states, num_states), dtype=np.int32)

  last_label = None
  for _, label in data:

    # Add transition
    if last_label is not None and label != last_label and \
       activities[last_label] not in excluded_activities and \
       activities[label] not in excluded_activities: 
      transitions_matrix[last_label - 1][label - 1] += 1
      
    if last_label is None or label != last_label:
      last_label = label
  
  print(transitions_matrix.shape)  
  return transitions_matrix


def draw_transitions_matrix(matrix):
  num_states = matrix.shape[0]
  activity_list = [activities[num] for num in range(1, num_states + 1)]
  
  fig, ax = plt.subplots(figsize=(12, 5))
  im, cbar = heatmap(matrix, activity_list, activity_list, ax=ax, cmap='Blues', cbarlabel='# Transitions')
  #texts = annotate_heatmap(im, valfmt='{x:d}')
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  fig.tight_layout()
  plt.savefig(output_image_path)
  

# def main():
#   train_data = read_data(train_path)
#   val_data = read_data(val_path)
#   test_data = read_data(test_path)
  
#   all_data = train_data + val_data + test_data
#   all_data = sorted(all_data, key=lambda x: x[0])

#   transitions_matrix = calculate_transitions_matrix(all_data)
#   draw_transitions_matrix(transitions_matrix)



def main():

  transitions_matrix = np.array([[0.716,0.279,0.005], [0.259,0.589,0.151],[0.019,0.234,0.747]])
  draw_transitions_matrix(transitions_matrix)




if __name__ == '__main__':
  main()
