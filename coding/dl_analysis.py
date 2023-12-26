# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_no = pd.read_csv('ptbdb_normal.csv')
data_ab = pd.read_csv('ptbdb_abnormal.csv')

data_no = np.array(data_no)
data_ab = np.array(data_ab)


# print(data_no.shape)      # (4045, 188)        
# print(data_ab.shape)      # (10505, 188)

# 데이터 갯수가 안맞는다.

nTrain = 3000
nTest = 1000

x_train = np.concatenate((data_no[:nTrain, :], data_ab[:nTrain, :]), 0)
y_train = np.concatenate((np.zeros(nTrain,), np.ones(nTrain,)), 0)          # normal인 경우 0을 넣고 abnormal인 경우에는 1을 넣는다
x_test = np.concatenate((data_no[nTrain:nTrain+nTest, :], data_ab[nTrain:nTrain+nTest, :]), 0)
y_test = np.concatenate((np.zeros(nTest,), np.ones(nTest,)), 0)

print(x_train.shape)
print(y_train.shape)

from tensorflow.keras.utils import to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers


model = Sequential()
model.add(layers.Conv1D(filters=16, kernel_size=3, input_shape=(x_train.shape[1], 1), activation='relu'))
model.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=3, strides=2))
model.add(layers.Conv1D(filters=16, kernel_size=3, input_shape=(x_train.shape[1], 1), activation='relu'))
model.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(layers.LSTM(16))
model.add(layers.Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.01), 
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2)

o = model.predict(x_test)
print(o)
# 0인지 1인지만 알면된다.
o = np.argmax(o, 1)
# 이제 y_test하고 비교
y_test = np.argmax(y_test, 1)
# 이렇게 하면 0.9275쯤 나온다. 92.75% 정도 나름 정확한 예측이 가능하다.
print(sum(np.equal(y_test, o)) / len(y_test))

