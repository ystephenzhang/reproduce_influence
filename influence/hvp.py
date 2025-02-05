import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.autograd.functional import hvp, hessian
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
from torch.func import functional_call

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import Manager

def calculate_grad_L(idx, model, dataset, bsize=4, graph=False):
    '''
    Calculate gradient of cross entropy loss w.r.t. model's parameter on the example given by idx.
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
        grad_this = torch.autograd.grad(loss, model.parameters(), create_graph=graph)
        grad_this = torch.cat((grad_this[0].flatten(), grad_this[1].flatten()))
        
        grad.append(grad_this)
    
    return grad

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

def inverse_hvp(train_dataset, model, v, t=5000, r=2, return_eig=False):
    '''
    Using the stochastic estimation method to calculate the product of an inverse Heissian and a vector.
    Three implementations are provided to compare speed/work as groud truth; implementation B is eventually
    selected, and the rest are commented.
    input:
        train_dataset - torch dataset used for stochastic estimaton of inv. H
        v - the vector to perform production with
        t - number of iterations during each approximation
        r - number of approximation to average across
        return_eig - implementation A allows for analysis of Hessian estimation's eigenvalues to study convergence issues.fun
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
            ''' 
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
            
            #C: Use hand-implemented hvp calculator
            '''
            hessian_vec = hvp_approx(model, y, x, product)
            '''
            hessian_vec = hessian_vec + 1e-4 * product
            old_product = product
            product = v + (old_product - hessian_vec) / 10
            print(f"Magnitude of change:{torch.norm(product - old_product):.2f}, Magnitude of product:{torch.norm(product):2f}")
        if Hv is None:
            Hv = product / 10
        else:
            Hv = Hv + product / 10
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
        criterion = nn.CrossEntropyLoss(reduction = 'mean')
        return criterion(outputs, y)        
    H = hessian(hessian_wrapper, params)
    return H

def actual_H_serial(train_dataset, model):
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

def worker(rank, world_size, return_list, train_dataset, model):
    '''
    Worker function for parrallelled oracle Hessian computation.
    '''
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=1024)

        #data: input, label, model
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    for data in tqdm(dataloader, desc=f"Rank {rank} processing"):
        input = torch.stack([x.view(-1) for x in data[0]]).to(device)
        label = data[1].to(device)
        result = calculate_sample_H(model, input, label)
        return_list.append(result.cpu())
    print(f"Rank {rank} completed")

    dist.destroy_process_group()

def actual_H_with_distributed_parallel(train_dataset, model):
    '''
    Entrance to calculating oracle Hessian.
    ''' 
    os.environ["MASTER_ADDR"] = "127.0.0.1"  
    os.environ["MASTER_PORT"] = "12355" 

    world_size = torch.cuda.device_count()
    manager = Manager()
    return_list = manager.list()
    mp.spawn(worker, args=(world_size, return_list, train_dataset, model), nprocs=world_size, join=True)
    
    n = len(return_list)
    H = torch.stack(list(return_list))
    H = torch.mean(H, dim = 0)
    print("completed, length:", n, "resulting size: ", H.shape)
    return H

    

