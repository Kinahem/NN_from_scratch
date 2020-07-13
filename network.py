import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as io
from tqdm import trange
from PIL import Image
import try_layers


size_koef = 10
np.random.seed(42)

def load_data(file, length):
    raw = io.loadmat(file)
    all_X = raw['X']
    all_y = raw['y']
    all_X = np.moveaxis(all_X, [3], [0])
    all_y = all_y.flatten()
    all_y[all_y == 10] = 0
    samples = np.random.choice(np.arange(all_X.shape[0]), 
                                length, replace=False)
    return all_X[samples].astype(np.float32), all_y[samples] #no float, we need int

def prepare_X(X):
    flat_X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
    mean_X = np.mean(flat_X, axis = 0)
    flat_X -= mean_X
    return flat_X

def split_train(X, y, num_val):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[:-num_val] #all besides last num_val 
    val_indices = indices[-num_val:] #last num_val
    train_X = X[train_indices]
    train_y = y[train_indices]
    val_X = X[val_indices]
    val_y = y[val_indices]
    return train_X, train_y, val_X, val_y

train_X, train_y = load_data(os.path.join("data", "train_32x32.mat"), 1000 * size_koef)
test_X, test_y = load_data(os.path.join("data", "test_32x32.mat"), 100 * size_koef)
train_X = prepare_X(train_X)
test_X = prepare_X(test_X)
train_X, train_y, val_X, val_y = split_train(train_X, train_y, 100 * size_koef)

# img = Image.fromarray(X[84], 'RGB')
# img.show()

network = []
network.append(try_layers.Dense(train_X.shape[1],100))
network.append(try_layers.ReLU())
network.append(try_layers.Dense(100,10))

def forward(network, X):
    activations = []
    input = X
    for l in network:
        activations.append(l.forward(input))
        input = activations[-1]
    
    assert len(activations) == len(network)
    return activations

def backward(network, grad, res):
    input = grad
    for l_index in range(len(network))[::-1]:
        layer = network[l_index]
        input = layer.backward(res[l_index], input)

def predict(network, X):
    # Compute network predictions. Returning indices of largest Logit probability
    res = forward(network,X)[-1]
    return res.argmax(axis=1)  


for epoch in range(100):
    #samples = np.random.choice(np.arange(train_X.shape[0]), 90 * size_koef, replace=False)
    loss = []
    for batch in range(9 * size_koef):
        batch_X = train_X[batch*100:(batch+1)*100]
        batch_y = train_y[batch*100:(batch+1)*100]
        res = forward(network, batch_X)
        loss.append(try_layers.softmax_with_cross_entropy(res[-1], batch_y))
        res = [batch_X] + res
        loss_grad = try_layers.grad_softmax_with_cross_entropy(res[-1], batch_y)
        backward(network, loss_grad, res)
    if epoch % 10 * size_koef == 0:
        network[0].scale_rate(0.0000495)
        network[2].scale_rate(0.0000495)
        print("epoch: {}   loss: {}".format(epoch, np.mean(loss)))
        print("val_accuracy: ", np.mean(predict(network,val_X)==val_y))
        print("test_accuracy: ", np.mean(predict(network,test_X)==test_y))

pred = predict(network,test_X)
for i in range(10):
    print("test_y:  {}, predict: {}".format(test_y[i], pred[i]))