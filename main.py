from scripts.train import *
from influence.hvp import *
import matplotlib.pyplot as plt
import os, pickle

def load_model(idx=None, epoch=20):
    if os.path.exists("data/models/trained_without" + str(idx) + ".pth"):
        model = LogisticRegressionModel(28 * 28, 10)
        model.load_state_dict(torch.load("data/models/trained_without" + str(idx) + ".pth"))
    else:
        model = train(remove=idx, epoch=epoch)
    
    return model

def save_result(lst, dir):
    with open(dir, 'wb') as f:
        pickle.dump(lst, f)
    
def leave_one_out(train_idx, test_idx):
    _model = load_model()
    model = load_model(train_idx)
     
    _, test_dataset = prepare_mnist(remove=None)
    x = test_dataset[test_idx[0]][0].view(1, -1)
    y = torch.tensor([test_dataset[test_idx[0]][1]]) 
    return test_single(model, x, y) - test_single(_model, x, y)
    
def calculate_influence(train_idx, train_dataset, test_idx, test_dataset, model):
    '''
    Calculate I_{up, loss}(z, z_{test}) as described in paper.
    '''
    grad_test = calculate_grad_L([test_idx], model, test_dataset)
    #s_test = inverse_hvp(train_dataset, model, grad_test[0])
    s_test = inverse_hvp_with_oracle(grad_test[0], "hessian_matrix.pt")
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
    influence = [-x / len(train_data) for x in influence]
    retrained = [leave_one_out(x[0], test_idx) for x in influence]
    return influence, retrained

if __name__ == "__main__":
    influnce, retrained = experiment(4)
    predicted_loss = [x[1].detach().numpy() for x in influnce]
    actual_loss = [x.detach().numpy() for x in retrained]
    save_result(predicted_loss, 'data/assets/p.pkl')
    save_result(predicted_loss, 'data/assets/a.pkl')
    print(predicted_loss, actual_loss)
    plt.scatter(predicted_loss, actual_loss)
    plt.xlabel('influence')
    plt.ylabel('retrained')
    
    plt.savefig('test.png')

'''
if __name__ == "__main__":
    from torch.utils.data import Subset
    if os.path.exists('trained_without_None.pth'):
        model = LogisticRegressionModel(28 * 28, 10)
        model.load_state_dict(torch.load('trained_without_None.pth'))
    else:
        model = train(remove=None, epoch=50)
    
    model.eval()
    train_data, _ = prepare_mnist_dataset()
    H = actual_H_with_distributed_parallel(train_data, model)
    torch.save(H, 'hessian_matrix.pt')
'''