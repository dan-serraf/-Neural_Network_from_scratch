# IMPORTATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from module.modele import *
from module.loss import *
from module.utils import *

# Création donnée : 2 gaussiennes
datax, datay = gen_arti(centerx=1, centery=1, sigma=0.4, nbex=500, data_type=0, epsilon=0.1)
testx, testy = gen_arti(centerx=1, centery=1, sigma=0.4, nbex=500, data_type=0, epsilon=0.1)
datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))

# Constante
input = datax.shape[1]
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
    hidden = linear.forward(datax)
    
    #backward
    losses.append(loss.forward(datay, hidden).mean())
    loss_back = loss.backward(datay, hidden)
    delta_linear = linear.backward_delta(datax, loss_back)
    
    #mise a jour paramètres
    linear.backward_update_gradient(datax, loss_back)
    linear.update_parameters(gradient_step=gradient_step)
    linear.zero_grad()

#Prediction
def predict(x):
    hidden = linear.forward(x)
    return np.where(hidden >= 0.5,1, 0)
yhat = predict(testx)

# Plot courbe Loss
plt.figure()
plt.plot(np.arange(iteration),losses)
plt.title("Courbe loss : 2 gaussiennes")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

# Plot data frontiere
acc = np.where(testy == yhat,1,0).mean()
print("accuracy : ",acc)
plt.figure()
plot_frontiere(testx, predict, step=100)
plot_data(testx, testy.reshape(-1))
plt.title("accuracy = "+str(acc))
plt.show()