import numpy as np
import os

SAMPLE_LEN = 1024

def load_dataset():
    file_path = 'data/'
    file_list = os.listdir(file_path)
    file_list = sorted(file_list)
    # print(file_list)

    merged_x = np.empty((0, SAMPLE_LEN))
    merged_y = np.empty((0, ))
    for i in file_list:
        data = np.load(file_path+i)
        merged_x = np.concatenate((merged_x, data['x']), axis = 0)
        merged_y = np.concatenate((merged_y, data['y']), axis = 0)

    indices = np.arange(len(merged_x))
    seed_value = 42
    np.random.seed(seed_value)
    np.random.shuffle(indices)

    shuffled_x = merged_x[indices]
    shuffled_y = merged_y[indices]

    return shuffled_x, shuffled_y

def load_dataset_list():
    file_path = 'data/'
    file_list = os.listdir(file_path)
    file_list = sorted(file_list)

    x_list = []
    y_list = []
    
    seed_value = 42
    np.random.seed(seed_value)
    
    for i in file_list:
        data = np.load(file_path+i)
        data_x = data['x']
        data_y = data['y']
        
        indices = np.arange(len(data_x))
        np.random.shuffle(indices)
        
        data_x = data_x[indices]
        data_y = data_y[indices]
        
        x_list.append(data_x)
        y_list.append(data_y)

    return x_list, y_list, file_list

def concatenate_data(x, y):
    merged_x = np.empty((0, SAMPLE_LEN))
    merged_y = np.empty((0, ))
    
    for i in range(len(x)):
        merged_x = np.concatenate((merged_x, x[i]), axis = 0)
        merged_y = np.concatenate((merged_y, y[i]), axis = 0)
        
    return merged_x, merged_y