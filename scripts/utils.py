import torch

def count_eigenvalue_categories(matrix):
    eigenvalues = torch.linalg.eigvals(matrix).real  # 只取实部

    num_negative = (eigenvalues < 0).sum().item()      # 负特征值数量
    num_zero = (eigenvalues == 0).sum().item()         # 等于 0 的特征值数量
    num_between_0_and_1 = ((eigenvalues > 0) & (eigenvalues <= 1)).sum().item()  # 0 到 1 之间
    num_greater_than_1 = (eigenvalues > 1).sum().item()  # 大于 1

    return [num_negative, num_zero, num_between_0_and_1, num_greater_than_1]