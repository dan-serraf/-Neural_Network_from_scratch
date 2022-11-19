# IMPORTATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from module.modele import *
from module.loss import *
from module.utils import *

np.random.seed(0)

# Création donnée : 4 gaussiennes
datax, datay = gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=1, epsilon=0.1)
testx, testy = gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=1, epsilon=0.1)
datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))

# Constante
input = datax.shape[1]
hidden = 64
output = 1
iteration = 100
gradient_step = 1e-4

# Module linéaire et Loss
loss = MSELoss()
linear1 = Linear(input, hidden)
linear2 = Linear(hidden, output)
tanh = TanH()
sigmoid = Sigmoid()

# Descente gradient 
losses = []
for _ in range(iteration):
    
    #forward
    hidden_linear1 = linear1.forward(datax)    
    hidden_tanh = tanh.forward(hidden_linear1)
    hidden_linear2 = linear2.forward(hidden_tanh)
    hidden_sigmoid = sigmoid.forward(hidden_linear2)
    
    
    #backward
    loss_back = loss.backward(datay, hidden_sigmoid)
    delta_sigmoid = sigmoid.backward_delta(hidden_linear2,loss_back)
    delta_linear2 = linear2.backward_delta(hidden_tanh,delta_sigmoid)
    delta_tanh = tanh.backward_delta(hidden_linear1,delta_linear2)
    delta_linear1 = linear1.backward_delta(datax,delta_tanh)
    losses.append(loss.forward(datay,hidden_sigmoid))
    
    #mise a jour paramètres
    linear2.backward_update_gradient(hidden_tanh, delta_sigmoid)
    linear1.backward_update_gradient(datax, delta_tanh)    
    linear2.update_parameters(gradient_step = gradient_step)
    linear1.update_parameters(gradient_step = gradient_step)
    linear2.zero_grad()
    linear1.zero_grad()
    

#Prediction
def predict(x):
    hidden = linear1.forward(x)
    hidden = tanh.forward(hidden)
    hidden = linear2.forward(hidden)
    hidden = sigmoid.forward(hidden)  
    return np.where(hidden >= 0.5,1,0)
yhat = predict(testx)

# Plot data frontiere
acc = np.where(testy == yhat,1,0).mean()
plt.figure()
print("accuracy : ",acc)
plot_frontiere(testx, predict, step=100)
plot_data(testx, testy.reshape(-1))
plt.title("accuracy = "+str(acc))
plt.show()