import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from helper import gen_I_Q_Amp_mat, gen_I_Q_Amp_Angle_mat
# import raw data
from load_dataset import load_dataset, load_dataset_list, concatenate_data

def prepare_train_dataset(dim = 3):
    val_idx = [2, 13]
    
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using:", device)
    
    x, y, l = load_dataset_list()
    
    print(f"validate with {[l[i] for i in val_idx]}")

    # print(len(x), len(y))

    # 3 CIR_data_NLOS_2_2.npz
    # 5 CIR_data_LOS_3_2.npz
    
    # for idx, name in enumerate(l):
    #     print(idx, name)
        
    def normalize_x(x):
        x_amp = np.abs(x)
        x_span = np.max(x_amp, axis=1) - np.min(x_amp, axis=1)

        x_norm = x.T/x_span.T
        x_norm = x_norm.T
        return x_norm

    x_norm = []
    for i in x:
        x_norm.append(normalize_x(i))
        
    x = x_norm

    x_prior = [data for idx, data in enumerate(x) if idx not in val_idx]
    y_prior = [data for idx, data in enumerate(y) if idx not in val_idx]
    x_post = [data for idx, data in enumerate(x) if idx in val_idx]
    y_post = [data for idx, data in enumerate(y) if idx in val_idx]
    x_train, y_train = concatenate_data(x_prior, y_prior)
    x_val, y_val = concatenate_data(x_post, y_post)

    total_len = len(y)
    
    if dim==1:
        x_train = np.abs(x_train)
        x_val = np.abs(x_val)
    elif dim==3:
        x_train = gen_I_Q_Amp_mat(x_train)
        x_val = gen_I_Q_Amp_mat(x_val)
    elif dim==4:
        x_train = gen_I_Q_Amp_Angle_mat(x_train)
        x_val = gen_I_Q_Amp_Angle_mat(x_val)

    # Convert numpy arrays to PyTorch tensors (use torch.LongTensor for indices)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)  # Change dtype to long
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)  # Change dtype to long
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)

    # Create PyTorch DataLoader for training and validation
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader
    