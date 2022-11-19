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

# Ajout bruit
p = 0.8
X_train_noise = add_noise(X_train,p=p)
X_test_noise = add_noise(X_test,p=p) 

# Constante
input = X_train.shape[1]
hidden = 100 # image 10 X 10
output = 144 # image 12 X 12
iteration = 5000
gradient_step = 1e-4
batchsize = 100

# Module linéaire et Loss
linear1 = Linear(input, hidden)
linear2 = Linear(hidden, output)
linear3 = Linear(output, hidden)
linear3._parameters = linear2._parameters.T
linear4 = Linear(hidden, input)
linear4._parameters = linear1._parameters.T
loss = BCELoss()

#AutoEncodeur
Encodeur = [linear1,TanH(),linear2,TanH()]
Decodeur = [linear3,TanH(),linear4,Sigmoid()]
AutoEncodeur = Sequentiel(Encodeur + Decodeur)

# Optimiseur
opt = Optim(AutoEncodeur,loss,eps=gradient_step)

# Descente gradient
mean, std = opt.SGD(X_train,X_train,batchsize,iteration,early_stop=100)

# Plot courbe Loss
plt.figure()
plt.title("Courbe loss : mnist auto encodeur")
plt.plot(mean)
plt.plot(std)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend(('Mean', 'std'))
plt.show()

reconstructed = opt._net.predict(X_test_noise)

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
plt.title("Courbe loss : mnist auto encodeur reconstruit")
plt.plot(mean)
plt.plot(std)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend(('Mean', 'std'))
plt.show()

# Accuracy
print("accuracy : ",opt.score(reconstructed,y_test))
