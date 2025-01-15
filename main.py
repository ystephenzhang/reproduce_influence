from scripts.train import *
from influence.hvp import *
from scripts.retraining import *
import matplotlib.pyplot as plt
import os, pickle

def calculate_influence(train_idx, train_dataset, test_idx, test_dataset, model):
    '''
    Calculate I_{up, loss}(z, z_{test}) as described in paper.
    '''
    grad_test = calculate_grad_L([test_idx], model, test_dataset)
    s_test = inverse_hvp(train_dataset, model, grad_test[0], return_eig=True)
    #s_test = inverse_hvp_with_oracle(grad_test[0], "data/assets/hessian_matrix.pt")
    grad_train = calculate_grad_L(train_idx, model, train_dataset)
    
    influence = [-torch.dot(s_test, x) for x in grad_train]
    return influence

def experiment(n=5):
    model = load_model()
    
    model.eval()
    train_data, test_data = prepare_mnist_dataset()
    test_idx = torch.randint(0, len(test_data), (1,))
    
    influence = calculate_influence(range(len(train_data)), train_data, test_idx, test_data, model)
    influence = sorted(enumerate(influence), key=lambda x: x[1], reverse=True)[:n]
    influence = [- x[1] / len(train_data) for x in influence]
    #retrained = calculate_retrained([x[0] for x in influence], test_idx)
    retrained = [leave_one_out(x[0], test_idx) for x in influence]
    return influence, retrained

if __name__ == "__main__":
    influnce, retrained = experiment(10)
    predicted_loss = [x[1].detach().numpy() for x in influnce]
    actual_loss = [x.detach().numpy() for x in retrained]
    save_result(predicted_loss, 'data/assets/p.pkl')
    save_result(actual_loss, 'data/assets/a.pkl')
    print(predicted_loss, actual_loss)
    plt.scatter(predicted_loss, actual_loss)
    plt.xlabel('influence')
    plt.ylabel('retrained')
    
    plt.savefig('test.png')