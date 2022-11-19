# IMPORTATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from module.modele import *
from module.loss import *
from module.utils import *


np.random.seed(0)

# Création donnée : mnist
nom_fichier_train =  "./mnist_train.csv"
nom_fichier_test =  "./mnist_test.csv"
data_train =  pd.read_csv(nom_fichier_train).to_numpy()
data_test =  pd.read_csv(nom_fichier_test).to_numpy()
X_train,y_train = data_train[:,1:].astype('float32') , data_train[:,0]
X_test, y_test = data_test[:,1:].astype('float32') , data_test[:,0]
X_train /= 255
X_test /= 255
y_train = onehot(y_train)

# Constante
input = X_train.shape[1]
hidden1 = 256
hidden2 = 128
output = 10
iteration = 250
gradient_step = 1e-4
batchsize = 100

# Module linéaire et Loss
linear1 = Linear(input, hidden1)
linear2 = Linear(hidden1, hidden2)
linear3 = Linear(hidden2, output)
tanh = TanH()
loss = CElogSoftMax()

#Prediction
def predict(x):
    return np.argmax(x,axis=1)

# Sequentiel
net = Sequentiel([linear1,tanh,linear2,tanh,linear3],labels=predict)

# Optimiseur
opt = Optim(net,loss,eps=gradient_step)

# Descente gradient
mean, std = opt.SGD(X_train,y_train,batchsize,iteration,early_stop=50)

# Plot courbe Loss
plt.figure()
plt.title("Courbe loss : mnist multi-classe")
plt.plot(mean)
plt.plot(std)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend(('Mean', 'std'))
plt.show()
print("accuracy : ",opt.score(X_test,y_test))

# Plot data
affiche_prediction(X_test,y_test,opt._net,6,n=28)