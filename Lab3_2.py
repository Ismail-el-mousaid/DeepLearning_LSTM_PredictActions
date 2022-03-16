# source: https://colab.research.google.com/drive/18WiSw1K0BW3jOKO56vxn11Fo9IyOuRjh

# numpy pour le calcul scientifique, matplotlib pour les graphiques et pandas pour la manipulation de données
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Chargez l'ensemble de données d'entraînement avec les colonnes « Open » et « High » à utiliser dans notre modélisation.
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
dataset_train = pd.read_csv(url)
training_set = dataset_train.iloc[:, 1:2].values

#Jetons un coup d'œil aux cinq premières lignes de notre ensemble de données
dataset_train.head()

#Importez MinMaxScaler depuis scikit-learn pour mettre à l'échelle notre ensemble de données en nombres compris entre 0 et 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Nous voulons que nos données soient sous la forme d'un tableau 3D pour notre modèle LSTM. Tout d'abord, nous créons des données en 60 pas de temps et les convertissons en un tableau à l'aide de NumPy. Ensuite, nous convertissons les données en un tableau 3D avec des échantillons X_train, 60 horodatages et une fonctionnalité à chaque étape.
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Faire les importations nécessaires depuis keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

#Ajoutez une couche LSTM avec des couches d'abandon pour éviter le surajustement. Après cela, nous ajoutons une couche Dense qui spécifie une sortie d'une unité. Ensuite, nous compilons le modèle à l'aide de l'optimiseur Adam et définissons la perte en tant que Mean_squarred_error
model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(X_train,y_train,epochs=100,batch_size=32)

#Importez l'ensemble de test pour le modèle pour faire des prédictions
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
dataset_test = pd.read_csv(url)
real_stock_price = dataset_test.iloc[:, 1:2].values

#Avant de prédire les cours boursiers futurs, nous devons manipuler l'ensemble d'apprentissage ; nous fusionnons l'ensemble d'apprentissage et l'ensemble de test sur l'axe 0, définissons le pas de temps sur 60, utilisons minmaxscaler et remodelons l'ensemble de données comme précédemment. Après avoir fait des prédictions, nous utilisons inverse_transform pour récupérer les cours des actions dans un format lisible normal.
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Tracez nos cours boursiers prévus et le cours boursier réel
plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()

