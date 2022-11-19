# IMPORTATION
import numpy as np
import copy 



# CLASSES MODULES
class Module(object):
    def __init__(self):
        self._parameters = None
        self._param_grad = None
        self._bias_grad = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._param_grad
        self._bias -= gradient_step * self._bias_grad

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass


# # PARTIE 1 : LINEAIRE

class Linear(Module):
    def __init__(self, input, output):
        self._input = input
        self._output = output
        self._parameters = np.random.random((self._input, self._output)) - 0.5  # valeur entre -1 et 1
        self._bias = np.random.random((1, self._output)) - 0.5
        self.zero_grad()
        #         self._parameters = np.zeros((self._input, self._output)) # valeur entre -1 et 1


    def zero_grad(self):
        self._param_grad = np.zeros((self._input, self._output))
        self._bias_grad = np.zeros((1, self._output))

    def forward(self, X):
#         assert X.shape[1] == self._input
        
        return np.dot( X, self._parameters) + self._bias

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._param_grad
        self._bias -= gradient_step * self._bias_grad

    def backward_update_gradient(self, input, delta):
#         assert input.shape[1] == self._input
#         assert delta.shape[1] == self._output
#         assert delta.shape[0] == input.shape[0] 
        
        self._param_grad += np.dot( input.T, delta )
        self._bias_grad += np.sum(delta, axis=0)

    def backward_delta(self, input, delta):
#         assert input.shape[1] == self._input
#         assert delta.shape[1] == self._output

        return np.dot( delta, self._parameters.T )


# PARTIE 2 : NON-LINEAIRE

class TanH(Module):
        
    def forward(self, X):
        return np.tanh(X)

    def backward_delta(self, input, delta):
        return ( 1 - np.tanh(input) ** 2 ) * delta
    
    def update_parameters(self, gradient_step=1e-3):
        pass


class Sigmoid(Module):
    
    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward_delta(self, input, delta):
        val = 1 / (1 + np.exp(-input) )
        return delta * ( val * ( 1 - val) )
    
    def update_parameters(self, gradient_step=1e-3):
        pass

# PARTIE 3 : ENCAPSULAGE
class Sequentiel:
   
    def __init__(self, modules,labels=None):
#         assert len(modules) > 0

        self._modules = modules
        self._labels = labels

    def forward(self, X): 
        list_forwards = [ self._modules[0].forward(X) ]
        for i in range(1, len(self._modules)):
            list_forwards.append( self._modules[i].forward( list_forwards[-1] ) )
       
        return list_forwards
    
    def backward_delta(self, list_forwards, delta):
        list_deltas =  [ delta ]
        for i in range(len(self._modules) - 1, 0, -1):
            self._modules[i].backward_update_gradient(list_forwards[i-1], list_deltas[-1])
            list_deltas.append( self._modules[i].backward_delta( list_forwards[i-1], list_deltas[-1] ) )
        
        return list_deltas

    def update_parameters(self, gradient_step=1e-3):
        for module in self._modules :
            module.update_parameters(gradient_step=gradient_step)
            module.zero_grad()
    
    def predict(self, x):
        return self._labels(self.forward(x)[-1]) if self._labels != None else self.forward(x)[-1]

    
class Optim:
    
    def __init__(self, net, loss, eps):
        
        self._net = net
        self._eps = eps
        self._loss = loss
        
        
    def step(self, batch_x, batch_y):
        
        list_forwards = self._net.forward( batch_x )
        loss = self._loss.forward(batch_y,list_forwards[-1])
        delta = self._loss.backward(batch_y,list_forwards[-1])
        list_deltas = self._net.backward_delta(list_forwards, delta)
        self._net.update_parameters(self._eps)
        return loss
      

    def SGD(self, X, Y, batch_size, epoch=10,early_stop=100):
#         assert len(X) == len(Y)
        
        # Mélange les indices
        indices = np.random.permutation(len(X))
        X2,Y2 = X.copy()[indices],Y.copy()[indices]
        
        #Création des listes batch
        range_X,range_Y = range(0, len(X2), batch_size),range(0, len(Y2), batch_size)
        
        batch_X = [X2[i:i + batch_size] for i in range_X]
        batch_Y = [Y2[i:i + batch_size] for i in range_Y]
        
        #Initialisations variables       
        list_mean,list_std = [],[]
        min_loss = float("inf")
        best_epoch,stop = 0,0
        best_model = self._net
        
        # Descente gradient 
        for epochs in range(epoch):
            
            list_temps = np.array([np.asarray(self.step(x, y)).mean() for x,y in zip(batch_X, batch_Y) ])
            loss_mean,loss_std = list_temps.mean(),list_temps.std()
            stop += 1
            
            if loss_mean < min_loss :
                stop, best_epoch, min_loss, best_model = 0, epochs, loss_mean, copy.deepcopy(self._net)
              
            if stop == early_stop :
                print("Early stop, best epoch : ",best_epoch)
                break
                
            list_mean.append(loss_mean)
            list_std.append(loss_std)
            
        self._net = best_model
        return list_mean, list_std
                
    def score(self,x,y):
        return np.where(y == self._net.predict(x),1,0).mean()


# PARTIE 4 : multi-classe

class Softmax(Module):

    def forward(self, X):
        exp = np.exp(X)
        return exp / np.sum(exp, axis=1).reshape((-1, 1))

    def backward_delta(self, input, delta):
        exp = np.exp(X)
        val = exp / np.sum(exp, axis=1).reshape((-1, 1))
        return delta * (val * (1 - val))

    def update_parameters(self, gradient_step=1e-3):
        pass
    
 

# PARTIE 5 : Auto-Encodeur


class AutoEncodeur:
    
     def __init__(self,encodeur,decodeur):
        self.encodeur = encodeur
        self.decodeur = decodeur
        
        self.autoEncodeur = Sequentiel(encodeur + decodeur)


# PARTIE 6 : Convolution

class ReLu(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return np.maximum(X, 0)

    def backward_delta(self, input, delta):
        return np.where(input>0, 1, 0) * delta

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=0.001):
        pass


class Conv1D(Module):

    def __init__(self, k_size, chan_in, chan_out, stride=1, biais=True):
        self._parameters = None
        self._gradient = None
        self.k_size=k_size
        self.chan_in=chan_in
        self.chan_out=chan_out
        self.stride=stride
        bound=1 / np.sqrt(chan_in*k_size)
       
        self._parameters = np.random.uniform(-bound, bound, (k_size,chan_in,chan_out))
        self._gradient=np.zeros(self._parameters.shape)
        self.biais = biais
        if(self.biais):
           
            self._biais=np.random.uniform(-bound, bound, chan_out)
            self._gradbiais = np.zeros((chan_out))

    def zero_grad(self):
        self._gradient=np.zeros(self._gradient.shape)
        if (self.biais):
            self._gradbiais = np.zeros(self._gradbiais.shape)

    def forward(self, X):
        size = ((X.shape[1] - self.k_size) // self.stride) + 1
        output=np.array([(X[:, i: i + self.k_size, :].reshape(X.shape[0], -1)) @ (self._parameters.reshape(-1, self.chan_out)) \
                         for i in range(0,size,self.stride)])
        if (self.biais):
            output+=self._biais
        self._forward=output.transpose(1,0,2)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient
        if self.biais:
            self._biais -= gradient_step * self._gradbiais

    def backward_update_gradient(self, input, delta):
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        output = np.array([ (delta[:,i,:].T) @ (input[:, i: i + self.k_size, :].reshape(input.shape[0], -1))  \
                           for i in range(0, size, self.stride)])
        self._gradient=np.sum(output,axis=0).T.reshape(self._gradient.shape)/delta.shape[0]

        if self.biais:
            self._gradbiais=delta.mean((0,1))

    def backward_delta(self, input, delta):
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        outPut = np.zeros(input.shape)
        for i in range(0, size, self.stride):
            outPut[:,i:i+self.k_size,:] += ((delta[:, i, :]) @ (self._parameters.reshape(-1,self.chan_out).T)).reshape(input.shape[0],self.k_size,self.chan_in)
        self._delta= outPut
        return self._delta



class Flatten(Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, X):
        self._forward = X.reshape(X.shape[0], -1)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        self._delta = delta.reshape(input.shape)
        return self._delta


class MaxPool1D(Module):

    def __init__(self, k_size=3, stride=1):
        super(MaxPool1D, self).__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        size = ((X.shape[1] - self.k_size) // self.stride) + 1
        outPut = np.zeros((X.shape[0], size, X.shape[2]))
        for i in range(0, size, self.stride):
            outPut[:,i,:]=np.max(X[:,i:i+self.k_size,:],axis=1)
        self._forward=outPut
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        outPut=np.zeros(input.shape)
        batch=input.shape[0]
        chan_in=input.shape[2]
        for i in range(0,size,self.stride):
            indexes_argmax = np.argmax(input[:, i:i+self.k_size,:], axis=1) + i
            outPut[np.repeat(range(batch),chan_in),indexes_argmax.flatten(),list(range(chan_in))*batch]=delta[:,i,:].reshape(-1)
        self._delta=outPut
        return self._delta

class AvgPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self._k_size = k_size
        self._stride = stride
        self.idx = []

    def forward(self, X):
        batch, length, chan_in = X.shape
        res = np.zeros((batch, (length - self._k_size) // self._stride + 1, chan_in))

        for ind_x in range(batch):
                for c in range(chan_in):
                    ind_res = 0
                    for i in range(0, length, self._stride):
                        if (i + self._k_size) > length:
                            break
                        res[ind_x, ind_res, c] = np.mean(X[ind_x,i:i+self._k_size,c])
        
                        ind_res += 1

        return res
    
    def backward_delta(self, input, delta):
        batch, length, chan_in = input.shape
        res = np.zeros((batch, length, chan_in))
        
        for ind_x in range(batch):
            for c in range(chan_in):
                ind = 0
                for i in range(0, length, self._stride):
                    if (i + self._k_size) < length:
                        indmax = np.argmax(input[ind_x,i:i+self._k_size,c])
                        res[ind_x, i+self._k_size, c] += delta[ind_x, ind, c]/self._k_size
                    ind += 1
        
        return res
     

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=0.001):
        pass