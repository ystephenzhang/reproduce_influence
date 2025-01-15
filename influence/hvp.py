import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd.functional import hvp, hessian
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import functional_call
from influence.reference import hessian_vector_product

from scripts.utils import *
def calculate_grad_L(idx, model, dataset, bsize=4, graph=False):
    '''
    input:
        idx - list of target indices
        model - logistic model
    output:
        grad - list of gradient of loss
    '''
    criterion = nn.CrossEntropyLoss()
    grad = []
    for i in tqdm(range(int(np.ceil(len(idx) / bsize))), desc="Calculating gradient of loss"):
        start = i * bsize
        end = min((i + 1) * bsize, len(idx))
        if not isinstance(idx[0], int):
            idx = [x.item() for x in idx]
        target = [dataset[i] for i in idx[start:end]]
        # print(len(target), target[0][1])
        x = torch.stack([x[0] for x in target]) 
        y = torch.stack([torch.tensor(x[1]) for x in target]) 
    
        model.zero_grad()
        pred = model(x.view(-1, 784))
        loss = criterion(pred, y)
        grad_this = torch.autograd.grad(loss, model.parameters(), create_graph=graph, retain_graph=True)
        grad_this = torch.cat((grad_this[0].flatten(), grad_this[1].flatten()))
        
        grad.append(grad_this)
    
    return grad

def _hvp_approx(ys, xs, v):
    """Multiply the Hessian of `ys` wrt `xs` by `v` in PyTorch.
    Args:
        ys: A scalar value, or a tensor to be summed to yield a scalar.
        xs: A list of tensors that we should construct the Hessian over.
        v: A list of tensors, with the same shapes as xs, that we want to
           multiply by the Hessian.
    Returns:
        A list of tensors (or if the list would be length 1, a single tensor)
        containing the product between the Hessian and `v`.
    """
    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop: Compute the first derivatives (gradients)
    grads = torch.autograd.grad(ys, xs, create_graph=True)

    # Ensure the length of grads matches xs
    assert len(grads) == length

    # Compute the elementwise product of the gradients and the vectors v
    elemwise_products = [
        grad_elem * v_elem for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]

    # Second backprop: Compute the second derivatives (gradients of the elementwise products)
    grads_with_none = torch.autograd.grad(elemwise_products, xs)

    # Replace None with zeros if there are any None gradients
    return_grads = [
        grad_elem if grad_elem is not None else torch.zeros_like(x)
        for x, grad_elem in zip(xs, grads_with_none)
    ]

    return return_grads

def compute_loss(model, x, y):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    outputs = model(x)
    loss = criterion(outputs, y)
    return loss

def hvp_approx(model, y, x, v):
    # Compute gradient of loss
    loss = compute_loss(model, x, y)
    grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    print("first backprop", grad[0].shape)
    grad = parameters_to_vector(grad)
    print("to vector", grad.shape)
    with torch.no_grad():
        elemwise_products = [
            grad_elem * v_elem for grad_elem, v_elem in zip(grad, v) if grad_elem is not None
        ]

    # Second backprop: Compute the second derivatives (gradients of the elementwise products)
    grad2 = torch.autograd.grad(elemwise_products, model.parameters())
    print("second backprop", grad2.shape, grad2[0].shape)
    hvp = parameters_to_vector(grad2)
    print("to vector", hvp.shape)
    return hvp

    grad_vector_product = torch.sum(grad * v)
    
    # Compute second derivative
    grad2 = torch.autograd.grad(grad_vector_product, model.parameters())
    hvp = parameters_to_vector(grad2)
    
    return hvp

def inverse_hvp(train_dataset, model, v, t=50, r=1, return_eig=False):
    '''
    Using the stochastic estimation method to calculate the product of an inverse Heissian and a vector.
    input:
        train_dataset - torch dataset used for stochastic estimaton of inv. H
        v - the vector to perform production with
    return:
        Hv - estimation of the inverse product
    '''
    eig = []
    Hv = None
    for i in tqdm(range(r), desc="Approximating inv. HVP, repetition"):
        product = v
        change = 0
        samples = torch.randint(0, len(train_dataset), (t,))
        for j in tqdm(range(t), desc=f"Iterating, stabilizing: {change:.2f}"):
            idx = samples[j]
            x = train_dataset[idx][0].view(1, -1)
            y = torch.tensor([train_dataset[idx][1]])
            
            #A: Calculate Hessian, then mat mul with the vector
            
            estimation = calculate_sample_H(model, x, y)
            if return_eig:
                eigen_cnt = count_eigenvalue_categories(estimation)
                eig.append(eigen_cnt)
            hessian_vec = torch.matmul(estimation, product)
            
            '''
            #B: use Pytorch hvp
            params = parameters_to_vector(model.parameters())
            param_shapes = {name: param.shape for name, param in model.named_parameters()}
            def vector_to_param_dict(vector, param_shapes):
                param_dict = {}
                offset = 0
                for name, shape in param_shapes.items():
                    numel = torch.prod(torch.tensor(shape))  # 参数的元素个数
                    param_dict[name] = vector[offset:offset + numel].view(shape)
                    offset += numel
                return param_dict
            def hessian_wrapper(new_params):
                new_params_dict = vector_to_param_dict(new_params, param_shapes)
                outputs = functional_call(model, new_params_dict, x)
                criterion = nn.CrossEntropyLoss()
                return criterion(outputs, y)
            result = hvp(hessian_wrapper, params, product)
            loss, hessian_vec = result
            '''
            #C: Use hand-implemented hvp calculator
            '''
            hessian_vec = hvp_approx(model, y, x, product)
            '''
            hessian_vec = hessian_vec + 1e-4 * product
            product = v + (product - hessian_vec) / 10
            print(torch.norm(product), torch.norm(hessian_vec))
        if Hv is None:
            Hv = product / 10
        else:
            Hv = Hv + product / 10
        #Hv = (Hv * i + product) / (i + 1)
        print(torch.norm(Hv))
    Hv = Hv / r

    if return_eig:
        eig = np.array(eig)
        np.savetxt('data/assets/eig.txt', eig, fmt='%d', delimiter=' ')
    
    return Hv
    
def calculate_sample_H(model, x, y):
    params = parameters_to_vector(model.parameters())
    param_shapes = {name: param.shape for name, param in model.named_parameters()}
    def vector_to_param_dict(vector, param_shapes):
        param_dict = {}
        offset = 0
        for name, shape in param_shapes.items():
            numel = torch.prod(torch.tensor(shape))  # 参数的元素个数
            param_dict[name] = vector[offset:offset + numel].view(shape)
            offset += numel
        return param_dict
    def hessian_wrapper(new_params):
        new_params_dict = vector_to_param_dict(new_params, param_shapes)
        outputs = functional_call(model, new_params_dict, x)
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, y)        
    H = hessian(hessian_wrapper, params)
    return H

def calculate_actual_H_serial(train_dataset, model):
    H = None
    samples = torch.randint(0, len(train_dataset), (1500, ))
    for j in tqdm(range(len(samples)), desc="Calculating ground truth H"):
        i = samples[j]
        x = train_dataset[i][0].view(1, -1)
        y = torch.tensor([train_dataset[i][1]])
        h = calculate_sample_H(model, x, y) 
        if H == None:
            H = h
        else:
            H = (H * i + h) / (i + 1)
    return H

def inverse_hvp_with_oracle(v, dir, t=20):
    H = torch.load(dir)
    product = v
    change = 0
    for j in tqdm(range(t), desc=f"Iterating, stabilizing: {change:.2f}"):
        old_product = product
        product = v + old_product - torch.matmul(H, old_product) / 10
        change = torch.norm(product - old_product)
        step = torch.norm(v - torch.matmul(H, old_product))
        print(change, step)
    return product / 10

