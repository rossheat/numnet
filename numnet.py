from tqdm import tqdm
import pandas as pd
import numpy as np

class Recorder:
    
    """ Record loss and metrics """
    
    def __init__(self):
        
        self.recorder = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'valid_acc'])
        
    def append(self, *args):
        
        """ Append row of loss and metric values """
        
        value_series = pd.Series([*args], index = self.recorder.columns)
        self.recorder = self.recorder.append(value_series, ignore_index=True)
        
    def history(self):
        
        """ Get recorder """
        
        return self.recorder
    
    def tail(self, n=5):
        
        """ Get last n rows of recorder """
        
        return self.recorder.tail(n)
    
    def plot_loss(self): 
        
        """ Plot training and validation losses """
        
        plot = self.recorder[['train_loss', 'valid_loss']].plot()
        plot.set_ylabel('Loss'); plot.set_xlabel('Epoch'); plot.set_title('Loss vs Epoch')
        return plot

class NumNet:
    
    """ Multi-layer classifier """
    
    def __init__(self, in_features, hidden_nodes, out_classes):
        
        self.W1, self.b1, self.W2, self.b2 = self._init_params(
            W1=(hidden_nodes, in_features), 
            b1=hidden_nodes, 
            W2=(out_classes, hidden_nodes),
            b2=out_classes
        )
    
        self.recorder = Recorder()

    def _train_valid_split(self, X, y, split):

        """ Split X and y into training and validation sets """

        assert len(X) == len(y), "Length of X and y must be equal"
        n_valid = int(len(X) * split)
        return X[n_valid:], y[n_valid:], X[:n_valid], y[:n_valid]

    def _init_params(self, W1, b1, W2, b2):

        """ Initialize weight and bias parameters """

        W1 = np.random.rand(*W1) - 0.5  
        b1 = np.random.rand(b1)  - 0.5
        W2 = np.random.rand(*W2) - 0.5
        b2 = np.random.rand(b2)  - 0.5
        return W1, b1, W2, b2
     
    def _linear(self, xb, W, b): 

        """ Perform a linear transformation """

        return xb@W.T + b 

    def _relu(self, xb): 

        """ Replace -ve values with zero """

        return np.maximum(xb, 0)

    def _softmax(self, x, axis=None):

        """ Convert activations to probability distribution """

        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def _forward(self, xb):

        """ Forward pass through network """

        Z1 = self._linear(xb, self.W1, self.b1)
        A1 = self._relu(Z1)
        Z2 = self._linear(A1, self.W2, self.b2)
        A2 = self._softmax(Z2, axis=1)
        return Z1, A1, Z2, A2

    def _backward(self, Z1, A1, Z2, A2, X, Y):

        """ Calculate the parameter gradients """

        dZ2 = A2.T - self._one_hot(Y)                  
        dW2 = 1 / len(X) * dZ2 @ A1            
        db2 = 1 / len(X) * np.sum(dZ2)
        dZ1 = self.W2.T @ dZ2 * self._relu_prime(Z1.T)
        dW1 = 1 / len(X) * dZ1 @ X      
        db1 = 1 / len(X) * np.sum(dZ1)              
        return dZ2, dW1, db1, dW2, db2

    def _step(self, dW1, db1, dW2, db2, lr):

        """ Update the model parameters """

        self.W1 -= lr * dW1                  
        self.b1 -= lr * db1                  
        self.W2 -= lr * dW2                  
        self.b2 -= lr * db2                  

    def _get_preds(self, probabilities): 

        """ Get class with largest probability """

        return probabilities.argmax(1)

    def _accuracy(self, probabilities, yb):

        """ Calculate accuracy """

        preds = self._get_preds(probabilities)
        return (preds == yb.T).astype(float).mean()

    def _relu_prime(self, Z): 

        """ Derivative of ReLU """

        return Z > 0

    def _one_hot(self, Y):

        """ One-hot encode targets """

        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    def _l1_loss(self, error):

        """ Calculate L1 loss (MAE) from error (y^ - y) """

        return np.abs(error).mean()
    
    def _train(self, X, y, epochs, lr):
        
        """ Train the model parameters """
        
        Z1, A1, Z2, A2 = self._forward(X)
        train_error, dW1, db1, dW2, db2 = self._backward(Z1, A1, Z2, A2, X, y) 
        self._step(dW1, db1, dW2, db2, lr)
        return self._l1_loss(train_error) 

    
    def _validate(self, X, y):
        
        """ Validate the model parameters """
        
        Z1, A1, Z2, A2 = self._forward(X) 
        valid_error, _, _, _, _ = self._backward(Z1, A1, Z2, A2, X, y)
        return self._l1_loss(valid_error), self._accuracy(A2, y)
    
    def _record(self, *args):
        
        """ Record the model parameters """
        
        value_series = pd.Series([*args], index = self.recorder.columns)
        self.recorder = self.recorder.append(value_series, ignore_index=True)

    def fit(self, X, y, epochs=50, lr=0.1, valid_pct=0.2):

        """ Train, validate, and record the model parameters """

        X_train, y_train, X_valid, y_valid = self._train_valid_split(X, y, valid_pct)

        for epoch in tqdm(range(epochs)):

            train_loss = self._train(X_train, y_train, epochs, lr)
            valid_loss, valid_acc = self._validate(X_valid, y_valid)
            self.recorder.append(epoch, train_loss, valid_loss, valid_acc)

        return self.recorder
    
    def _reshape(self, X):
        
        """ Add dimension to rank-1 array """
        
        return X.reshape(1, -1) if X.ndim == 1 else X

    def predict(self, X):

        """ Predict targets """

        X = self._reshape(X)
        _, _, _, A2 = self._forward(X)
        return self._get_preds(A2)
    
    def test(self, X, y):
        
        """ Evaluate model on test set """

        X = self._reshape(X)
        _, _, _, A2 = self._forward(X)
        return self._accuracy(A2, y)
    
    def parameters(self):
        
        """ Get model parameters """
        
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
