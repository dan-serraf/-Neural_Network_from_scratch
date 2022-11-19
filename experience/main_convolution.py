# IMPORTATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from module.modele import *
from module.loss import *
from module.utils import *


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns



# Load Data From USPS , directement pris depuis TME4
uspsdatatrain = "../data/USPS_train.txt"
uspsdatatest = "../data/USPS_test.txt"

alltrainx, alltrainy = load_usps(uspsdatatrain)
alltestx, alltesty = load_usps(uspsdatatest)

# taille couche
input = len(alltrainx[0])
output = len(np.unique(alltesty))
alltrainy_oneHot = onehot(alltrainy)
alltesty_oneHot = onehot(alltesty)

# Standardisation
scaler = StandardScaler()
alltrainx = scaler.fit_transform(alltrainx)
alltestx = scaler.fit_transform(alltestx)

alltrainx /= 2
alltestx /= 2


print(alltrainx.shape)

alltrainx = alltrainx.reshape(alltrainx.shape[0], alltrainx.shape[1], 1)
alltestx = alltestx.reshape(alltestx.shape[0], alltestx.shape[1], 1)

print(alltrainx.shape)


iteration = 1
gradient_step = 1e-3
batch_size = 50

l1 = Conv1D(3, 1, 32,biais=False)
l2 = AvgPool1D(2, 2)
l3 = Flatten()
l4 = Linear(12512, 100)
l5 = ReLu()
l6 = Linear(100, alltrainy_oneHot.shape[1])
l7 = Softmax()

#Prediction
def predict(x):
    return np.argmax(x,axis=1)

model = Sequentiel(l1, l2, l3, l4, l5, l6,labels=predict)
loss = CElogSoftMax()
opt = Optim(model , loss , eps = gradient_step)
mean, std = opt.SGD( alltrainx, alltrainy_oneHot, batch_size, epoch=batch_size, early_stop=100)
#list_loss = []


list_loss = mean

# Predection
predict = model.forward(alltrainx)
predict = np.argmax(predict, axis=1)

predict_test = model.forward(alltestx)
predict_test = np.argmax(predict_test, axis=1)

print(predict.shape)
print(alltrainy.shape)

print("Precision sur l'ensemble d'entrainement",((np.sum(np.where(predict == alltrainy, 1, 0)) / len(predict))*100),"%")
print("Precision sur l'ensemble de test",((np.sum(np.where(predict_test == alltesty, 1, 0)) / len(predict_test))*100),"%")



taux_train = ((np.argmax( opt.net.forward(alltrainx),axis = 1) == alltrainy).mean()*100)
taux_test = ((np.argmax( opt.net.forward(alltestx),axis = 1) == alltesty).mean()*100)
print("Taux de bonne classification en train : ",taux_train,"%")
print("Taux de bonne classification en test : ",taux_test,"%")

"""
AFFICHAGE DE LA LOSS
"""
plt.figure()
plt.xlabel("nombre d'iteration")
plt.ylabel("Erreur CE")
plt.title("Evolution de l'erreur")
plt.plot(list_loss,label="Erreur")
plt.legend()
plt.show()



# Confusion Matrix
plt.figure()
confusion = confusion_matrix(predict, alltrainy)
ax = sns.heatmap(confusion, annot=True, cmap='Blues')
ax.set_title(f"Matrice de confusion pour données USPS Train \ acc = {taux_train}%\n\n")
ax.set_xlabel('\nChiffre prédit')
ax.set_ylabel('Vrai chiffre ')
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(np.arange(10))
ax.yaxis.set_ticklabels(np.arange(10))
## Display the visualization of the Confusion Matrix.
plt.show()


# Confusion Matrix
plt.figure()
confusion = confusion_matrix(predict_test, alltesty)
ax = sns.heatmap(confusion, annot=True, cmap='Blues')
ax.set_title(f"Matrice de confusion pour données USPS Test \ acc = {taux_test}%\n\n")
ax.set_xlabel('\nChiffre prédit')
ax.set_ylabel('Vrai chiffre ')
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(np.arange(10))
ax.yaxis.set_ticklabels(np.arange(10))
## Display the visualization of the Confusion Matrix.
plt.show()
