import torch

def count_eigenvalue_categories(matrix):
    """
    输入: 一个 tensor 矩阵 (方阵)
    输出: 一个列表，包含 [负特征值数量, 等于0的特征值数量, 介于 0 到 1 之间的特征值数量, 大于 1 的特征值数量]
    """
    # 检查输入是否是方阵
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("输入必须是方阵！")

    # 计算特征值
    eigenvalues = torch.linalg.eigvals(matrix).real  # 只取实部

    # 统计特征值数量
    num_negative = (eigenvalues < 0).sum().item()      # 负特征值数量
    num_zero = (eigenvalues == 0).sum().item()         # 等于 0 的特征值数量
    num_between_0_and_1 = ((eigenvalues > 0) & (eigenvalues <= 1)).sum().item()  # 0 到 1 之间
    num_greater_than_1 = (eigenvalues > 1).sum().item()  # 大于 1

    # 返回统计结果列表
    return [num_negative, num_zero, num_between_0_and_1, num_greater_than_1]