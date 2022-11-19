# IMPORTATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from module.modele import *
from module.loss import *
from module.utils import *

np.random.seed(0)

# Création donnée : echiquier
datax, datay = gen_arti(centerx=1, centery=1, sigma=0.5, nbex=500, data_type=2, epsilon=0.1)
testx, testy = gen_arti(centerx=1, centery=1, sigma=0.5, nbex=500, data_type=2, epsilon=0.1)
datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))

# Constante
input = datax.shape[1]
hidden1 = 256
hidden2 = 128
hidden3 = 64
output = 1
iteration = 5000
gradient_step = 1e-3
batchsize=10

# Module linéaire et Loss
loss = MSELoss()
linear1 = Linear(input, hidden1)
linear2 = Linear(hidden1, hidden2)
linear3 = Linear(hidden2, hidden3)
linear4 = Linear(hidden3, output)
tanh = TanH()
sigmoid = Sigmoid()

#Prediction
def predict(x):
    return np.where(x >= 0.5,1, 0)

# Sequentiel
net = Sequentiel([linear1,tanh,
                  linear2,tanh,
                  linear3,tanh,
                  linear4,sigmoid],labels=predict)
# Optimiseur
opt = Optim(net,loss,eps= gradient_step)

# Descente gradient 
mean, std = opt.SGD(datax,datay,batchsize,iteration)

# Plot courbe Loss
plt.figure()
plt.title("Courbe loss : echiquier")
plt.plot(mean)
plt.plot(std)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend(('Mean', 'std'))
plt.show()

# Plot data frontiere
acc = opt.score(datax,datay)
print("accuracy : ",acc)
plt.figure()
plot_frontiere(datax, opt._net.predict, step=100)
plot_data(datax, datay.reshape(-1))
plt.title("accuracy = "+str(acc))
plt.show()