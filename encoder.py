import pandas as pd 
import numpy as np 
import glob
import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import UpSampling1D, Reshape, TimeDistributed, concatenate, Dense, BatchNormalization, Dropout,  Input, Flatten, Activation, Conv1D, Conv2D, LSTM, AveragePooling1D, AveragePooling2D
from data_set import data_set


print('Loading --> ')
X = data_set(load_n = -1, shuffle = True, return_joined = True, n_samples = 8192)
x_train, y_train, x_test, y_test = X.generate()
X = x_train[0]
print('Dataset shape(X)', X.shape)
print(np.sum(np.isnan(X)))
split = 0.8

## train test split

total = X.shape[0]

x_train = X[:int(split*total),:,:]
y_train = x_train.reshape(x_train.shape[0], -1)
x_test = X[int(split*total):,:,:]
y_test = x_test.reshape(x_test.shape[0], -1)

print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)

n_features = int(x_train.shape[2])
n_timesteps = int(x_train.shape[1])

def cnn_encoder(n_timesteps, n_features):
    drop = 0
    input_ = Input((n_timesteps, 12))
    x = AveragePooling1D(4)(input_)
    x = Conv1D(32, 3, padding = 'same', activation = 'relu', name = 'encoder1')(x)
    x = AveragePooling1D(2)(x)
    x = Conv1D(64, 3, padding = 'same', activation = 'relu', name = 'encoder2')(x)
    x = AveragePooling1D(2)(x)
    x = Conv1D(64, 3, padding = 'same', activation = 'relu', name = 'encoder3')(x)
    x = AveragePooling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'encoder4')(x)
    x = AveragePooling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'encoder5')(x)
    x = AveragePooling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'encoder6')(x)
    x = AveragePooling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'encoder7')(x)
    # x = AveragePooling1D(2)(x)
    # x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'encoder8')(x)
    # x = AveragePooling1D(2)(x)
    # x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'encoder9')(x)
    
    x = AveragePooling1D(2, name = 'encoded')(x)

    # x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'decoder10')(x)
    # x = UpSampling1D(2)(x)
    # x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'decoder9')(x)
    # x = UpSampling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'decoder8')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'decoder7')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'decoder6')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'decoder5')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'decoder4')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'decoder3')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(128, 3, padding = 'same', activation = 'relu', name = 'decoder2')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, padding = 'same', activation = 'relu', name = 'decoder1')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, padding = 'same', activation = 'relu', name = 'decoder0')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, padding = 'same', name = 'decoder-1')(x)
    x = Conv1D(12, 3, activation = 'sigmoid', padding = 'same', name = 'recover')(x)
    x = Flatten()(x)

    model = Model(inputs = input_, outputs = x)

    return model
    
model = cnn_encoder(n_timesteps, n_features)

epochs = 1000
batch_size = 32

model.compile(optimizer = 'adam', loss = 'mse')

print(model.summary())
earlystop = tensorflow.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 40, mode = 'min', restore_best_weights = True, verbose = 0)
hisotry = model.fit(x = x_train, y = y_train, validation_data = [x_test, y_test], shuffle = True, use_multiprocessing = True, workers = 12, verbose = 2, batch_size = batch_size, epochs = epochs, callbacks = [earlystop])

model.save('./deep-encoder.h5')

print('Saving encoder' + './deep-encoder.h5')