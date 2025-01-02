import torch.nn as nn
from torch.autograd.functional import hvp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import functional_call, hessian
import torch

def hvp_(model, x, y, v):
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
        criterion = nn.MSELoss()
        return criterion(outputs, y)
    result = hvp(hessian_wrapper, params, v)
    loss, hessian_vec = result 
    return hessian_vec

def compute_loss(model, x, y):
    criterion = nn.MSELoss()
    outputs = model(x)
    loss = criterion(outputs, y)
    return loss

def hessian_(model, x, y):
    loss = compute_loss(model, x, y)
    params = list(model.parameters())
    env_grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)

    hess_params = torch.zeros(2, 1, 2, 1)
    for i in range(env_grads[0].size(0)):
        for j in range(env_grads[0].size(1)):
            hess_params[i, j] = torch.autograd.grad(env_grads[0][i][j], params, retain_graph=True)[0]
    hess_params = hess_params.view(3, -1) 
    return hess_params

if __name__ == "__main__":
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(2, 1)  # 单层全连接

        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    print(list(model.parameters()))
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)  # 输入样本
    y = torch.tensor([1.0], requires_grad=True).view(1, -1)  # 目标值
    v = torch.tensor([3.0, 3.0, 3.0])
    
    #hvp_result = hvp_(model, x, y, v)
    hess = hessian_(model, x, y)
    print(hess)
    