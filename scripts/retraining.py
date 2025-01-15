from scripts.model import *
from scripts.train import *
import os, pickle

def load_model(idx=None, epoch=20):
    if os.path.exists("data/models/trained_without_" + str(idx) + ".pth"):
        model = LogisticRegressionModel(28 * 28, 10)
        model.load_state_dict(torch.load("data/models/trained_without_" + str(idx) + ".pth"))
    else:
        model = train(remove=idx, epoch=epoch)
    
    return model

def save_result(lst, dir):
    with open(dir, 'wb') as f:
        pickle.dump(lst, f)
    
def leave_one_out(train_idx, test_idx):
    _model = load_model()
    model = load_model(train_idx)
     
    _, test_dataset = prepare_mnist_dataset()
    x = test_dataset[test_idx[0]][0].view(1, -1)
    y = torch.tensor([test_dataset[test_idx[0]][1]]) 
    return test_single(model, x, y) - test_single(_model, x, y)

