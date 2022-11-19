 # IMPORTATION
import numpy as np



# CLASSES COÛTS
class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss(Loss):
    def forward(self, y, yhat):
        assert(y.shape == yhat.shape)

        return np.sum((y - yhat) ** 2,axis = 1)

    def backward(self, y, yhat):
        assert(y.shape == yhat.shape)

        return -2 * (y-yhat)

    
class CELoss(Loss):
    
    def forward(self, y, yhat):
#         assert(y.shape == yhat.shape)
        
        return 1 - np.sum(yhat * y, axis = 1)
    
    def backward(self, y, yhat):
#         assert(y.shape == yhat.shape)
        
        return yhat - y
    
    
class CElogSoftMax(Loss):
    
    def forward(self, y, yhat,eps = 1e-100):
#         assert(y.shape == yhat.shape)

        return np.log(np.sum(np.exp(yhat), axis=1) + eps) - np.sum(y * yhat,axis = 1)

    def backward(self, y, yhat):
#         assert(y.shape == yhat.shape)
         
        exp = np.exp(yhat)
        return exp / np.sum(exp, axis=1).reshape((-1,1)) - y  


class BCELoss(Loss):
    
    def forward(self, y, yhat,eps = 1e-100):
        assert(y.shape == yhat.shape)
        
        return - (y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))
    
    def backward(self, y, yhat,eps = 1e-100):
        assert(y.shape == yhat.shape)
        
        return ((1 - y) / (1 - yhat + eps)) - (y / yhat + eps)

