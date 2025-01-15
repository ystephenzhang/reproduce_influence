import torch

def count_eigenvalue_categories(matrix):
    """
    Count negative, zero, 0-to-1 and greater-than-1 eigenvalues.
    """
    eigenvalues = torch.linalg.eigvals(matrix).real  

    num_negative = (eigenvalues < 0).sum().item()      
    num_zero = (eigenvalues == 0).sum().item()         
    num_between_0_and_1 = ((eigenvalues > 0) & (eigenvalues <= 1)).sum().item()  
    num_greater_than_1 = (eigenvalues > 1).sum().item()  

    return [num_negative, num_zero, num_between_0_and_1, num_greater_than_1]