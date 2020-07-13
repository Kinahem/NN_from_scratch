import numpy as np
np.random.seed(42)

class Dense:
    def __init__(self, input_units, output_units, learning_rate=0.05):
        self.learning_rate = learning_rate
        self.W = np.random.normal(loc=0.0, scale = np.sqrt(0.1/(input_units+output_units)), size = (input_units,output_units))#np.zeros((input_units,output_units))
        self.B = np.random.normal(loc=0.0, scale = np.sqrt(0.1/(input_units+output_units)), size = (output_units))#np.zeros(output_units)

    def forward(self, X):
        return np.dot(X, self.W) + self.B

    def backward(self, input, grad_output):
        #
        grad_input = np.dot(grad_output, self.W.T)
        
        # compute gradient w.r.t. weights and biases
        grad_W = np.dot(input.T, grad_output)
        grad_B = grad_output.mean(axis=0)*input.shape[0]
        
        assert grad_W.shape == self.W.shape and grad_B.shape == self.B.shape
        
        # Here we perform a stochastic gradient descent step. 
        self.W = self.W - self.learning_rate * grad_W
        self.B = self.B - self.learning_rate * grad_B
        return grad_input

    def scale_rate(self, decrease):
        self.learning_rate -= decrease

class ReLU:
    def __init__(self):
        pass

    def forward(self, X):
        return np.maximum(0, X)

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad

def softmax_with_cross_entropy(predictions, y):
    stable_pred = np.copy(predictions)
    stable_pred -= np.max(stable_pred, axis = 1, keepdims=True)
    probs = np.exp(stable_pred)/np.sum(np.exp(stable_pred), axis = 1, keepdims=True)
    m = y.shape[0]
    return np.sum(-np.log(probs[range(m), y.T])) / m

def grad_softmax_with_cross_entropy(predictions, y):
    m = y.shape[0]
    grad = softmax(predictions)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

def softmax(predictions): #don't use
    stable_pred = np.copy(predictions)
    stable_pred -= np.max(stable_pred, axis = 1, keepdims=True)
    probs = np.exp(stable_pred)/np.sum(np.exp(stable_pred), axis = 1, keepdims=True)
    return probs

def cross_entropy(probs, y): #don't use
    m = y.shape[0]
    return np.sum(-np.log(probs[range(m), y.T])) / m