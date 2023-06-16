## Backpropagation for neural network sensitivity analysis

def affine_backward(dout, cache):
    x, w = cache
    dx = np.dot(dout, w.T)
    return dx
def relu_backward(dout, cache):
    x = cache
    dx = dout * np.where(x > 0, np.ones(x.shape), np.zeros(x.shape))
    return dx
def sigmoid_backward(cache):
    dx = cache*(1 - cache)
    return dx

## Data Normalization

def norm(test, train):
    mean_trans = train.mean().to_numpy().reshape((train.shape[1],1))*np.ones((1,test.shape[0]))
    mean = mean_trans.T
    std_trans = train.std().to_numpy().reshape((train.shape[1],1))*np.ones((1,test.shape[0]))
    std = std_trans.T
    transformed = np.divide(test.to_numpy() - mean, std)
    return pd.DataFrame(transformed, columns = train.columns)
