# IMPORTATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from module.modele import *
from module.loss import *
from module.utils import *

# Expérience 1 Linéaire : Régression 

np.random.seed(0) 
# Création donnée : Fonction affine ax+b
x,y = fonction_affine_bruit(40,8)

# Constante
input = x.shape[1]
output = 1
iteration = 100
gradient_step = 1e-4

# Module linéaire et Loss
loss = MSELoss()
linear = Linear(input, output)


# Descente gradient 
losses = []
for _ in range(iteration):
    #forward
    hidden = linear.forward(x)
    
    #backward
    losses.append(loss.forward(y, hidden).mean())
    loss_back = loss.backward(y, hidden)
    delta_linear = linear.backward_delta(x, loss_back)
    
    #mise a jour paramètres
    linear.backward_update_gradient(x, loss_back)
    linear.update_parameters(gradient_step=gradient_step)
    linear.zero_grad()

#Prediction
def predict(x):
    return linear.forward(x)
yhat = predict(x)

# Plot data
plt.figure()
plt.scatter(x,y,label="data",color='black')
plt.plot(x,yhat,color='red',label='predection')
for i in range(len(x)):
    plt.plot([x[i],x[i]],[y[i], yhat[i]], c="blue", linewidth=1) 
    
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Prediction ax + b ")
plt.show()

# Plot courbe Loss
plt.figure()
plt.plot(np.arange(iteration),losses)
plt.title("Courbe loss ax+b")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()