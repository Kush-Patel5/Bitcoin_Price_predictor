import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU

Training_data = pd.read_csv("BTC-USD Training Data.csv")
Open_training = Training_data["Open"]
High_training = Training_data["High"]
Low_training = Training_data["Low"]
Close_training = Training_data["Close"]

Testing_data = pd.read_csv("BTC-USD Testing Data.csv")
Open_testing = Testing_data["Open"]
High_testing = Testing_data["High"]
Low_testing = Testing_data["Low"]
Close_testing = Testing_data["Close"]

X_data = []
y_data = []
for i in range(len(Open_training) - 6):
    X_data.append([[Open_training[j], High_training[j], Low_training[j], Close_training[j]] for j in range(i, i + 5)])
    y_data.append([Open_training[i + 5], High_training[i + 5], Low_training[i + 5], Close_training[i + 5]])
X_data = np.array(X_data)
y_data = np.array(y_data)

model = Sequential()
model.add(tf.keras.layers.Flatten())
model.add(Dense(units=20, input_shape=(5, 4), activation="LeakyReLU"))
model.add(Dense(units=10, activation="LeakyReLU"))
model.add(Dense(units=4, activation="LeakyReLU"))
model.add(Dense(units=4, activation="LeakyReLU"))
model.compile(optimizer="adam",
                   loss="mean_absolute_error",
                   metrics=["accuracy"])
model.fit(X_data, y_data, epochs=250)

X_test = [[[Open_testing[t], High_testing[t], Low_testing[t], Close_testing[t]] for t in range(5)]]
prediction = model.predict(X_test)
print("These are the predictions for the Open, High, Low, and Close prices for 2/18/22 using data from 2/13/22 - 2/17/22:")
print("Open: " + str(prediction[0][0]))
print("High: " + str(prediction[0][1]))
print("Low: " + str(prediction[0][2]))
print("Close: " + str(prediction[0][3]))

Open_Accuracy = (1 - (abs(prediction[0][0] - Open_testing[5])/Open_testing[5]))
High_Accuracy = (1 - (abs(prediction[0][1] - High_testing[5])/High_testing[5]))
Low_Accuracy = (1 - (abs(prediction[0][2] - Low_testing[5])/Low_testing[5]))
Close_Accuracy = (1 - (abs(prediction[0][3] - Close_testing[5])/Close_testing[5]))
Accuracy = (Open_Accuracy + High_Accuracy + Low_Accuracy + Close_Accuracy)/4
print("Accuracy: " + str(round(Accuracy * 100, 2)))
input("Press enter to exit")