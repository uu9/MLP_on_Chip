import numpy as np

def gen_I_Q_Amp_mat(complex_matrix):
    
    # 提取实部和虚部作为 I 路和 Q 路
    real_part = np.real(complex_matrix)
    imaginary_part = np.imag(complex_matrix)

    # 计算幅度
    amplitude = np.abs(complex_matrix)

    # 创建 batch_size x 3 x 1024 的矩阵
    result_matrix = np.stack([real_part, imaginary_part, amplitude], axis=1)
    
    return result_matrix

def gen_I_Q_Amp_Angle_mat(complex_matrix):
    
    # 提取实部和虚部作为 I 路和 Q 路
    real_part = np.real(complex_matrix)
    imaginary_part = np.imag(complex_matrix)

    # 计算幅度
    amplitude = np.abs(complex_matrix)
    
    angle = np.angle(complex_matrix) / np.pi

    # 创建 batch_size x 3 x 1024 的矩阵
    result_matrix = np.stack([real_part, imaginary_part, amplitude, angle], axis=1)
    
    return result_matrix

def count_parameters(model):
    k = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Number of parameters: {k:.2f} M")
    return k

def count_parameters_bytes(model):
    k = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"Number of parameters: {k * 4:.2f} KB")