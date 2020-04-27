import pandas as pd 
import numpy as np 
import glob
import tensorflow.compat.v1 as tensorflow
# import tensorflow
tensorflow.disable_v2_behavior()
tensorflow.compat.v1.enable_eager_execution()
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import concatenate, Reshape, TimeDistributed, concatenate, Dense, BatchNormalization, Dropout,  Input, Flatten, Activation, Conv1D, Conv2D, LSTM, AveragePooling1D, AveragePooling2D
from data_set import data_set

print('Loading --> ')
data_set = data_set(load_n = -1, shuffle = True, return_joined = True, n_samples = 8192)
x_train, y_train, x_test, y_test = data_set.generate()
cat_features = data_set.get_cat_features()
n_labels = data_set.get_n_classes()
print('number of cat features: ', cat_features)
print('number of classes: ', n_labels)


def load_leads_model(trainable, drop, interdrop):
    print('loading joined model')
    loaded_model = tensorflow.keras.models.load_model('./deep-encoder.h5', compile = False)
    model = Sequential()
    done = False
    for l in loaded_model.layers:
        if not done:
            if l.name == 'encoded':
                done = True
            model.add(l)
            if l.name in ['average_pooling1d', 'average_pooling1d_1', 'average_pooling1d_2', 'average_pooling1d_3', 'average_pooling1d_4', 'average_pooling1d_5']:
                model.add(Dropout(interdrop))
    model.add(Flatten(name = 'encode-flatten'))
    model.add(Dropout(drop))
    for l in model.layers:
        l.trainable = trainable

    return model

def create_cat_model(n_features):
    input_ = Input((n_features), name = 'cat-input')
    x = Dense(32)(input_)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    model = Model(inputs = input_, outputs = x)
    return model

def baseline_model(cat_features, n_labels):

    print('create cat model')
    cat_model = create_cat_model(cat_features)
    
    cat_model_input = cat_model.input
    cat_model_output = cat_model.output
    model_li = load_leads_model(trainable = True, interdrop = 0, drop = 0)
    leads_model_inputs = [model_li.input]
    leads_model_outputs = [model_li.output]
    input_ = leads_model_inputs
    input_.extend([cat_model_input])
    output_ = leads_model_outputs
    output_.extend([cat_model_output])

    z = concatenate(output_)

    z = Dense(1024)(z)
    z = Activation('relu')(z)
    z = Dropout(0.5)(z)

    z = Dense(n_labels, activation = 'sigmoid')(z)

    model = Model(inputs = input_, outputs = z)

    return model

print('Generate Model')
model = baseline_model(cat_features, n_labels)

epochs = 1000
batch_size = 32

def tp(y_true, y_pred):
    sw = ~tensorflow.less(y_true, 0.5)
    acc = tensorflow.keras.metrics.binary_accuracy(y_true[sw], y_pred[sw])
    return acc

def tn(y_true, y_pred):
    sw = tensorflow.less(y_true, 0.5)
    acc = tensorflow.keras.metrics.binary_accuracy(y_true[sw], y_pred[sw])
    return acc


model.compile(optimizer = tensorflow.keras.optimizers.Adam(0.001), loss = 'binary_crossentropy', metrics = ['accuracy', tp, tn, tfa.metrics.FBetaScore(num_classes = 9, average = 'weighted', beta = 2.0)])

weight_for_0 = 0.05
weight_for_1 = 1
# class_weight = {0: weight_for_0, 1: weight_for_1}

print(model.summary())
earlystop = tensorflow.keras.callbacks.EarlyStopping(monitor = 'val_fbeta_score', patience = 40, mode = 'max', restore_best_weights = True, verbose = 0)
hisotry = model.fit(x = x_train, y = y_train, shuffle = True, verbose = 2, validation_data = [x_test, y_test], batch_size = batch_size, epochs = epochs, callbacks = [earlystop])
model.evaluate(x_test, y_test)
model.save('./baseline.h5')

print('Finished.')

pred = model.predict(x_test)
eval_ = y_test
print(pred.shape)

score_li = []
zero_counts = []
one_counts = []
one_percent = []
true_pos = []
false_pos = []
true_neg = []
false_neg = []
categories = np.array(['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE'])
from sklearn.metrics import accuracy_score, confusion_matrix

for col in range(pred.shape[1]):
    class_pred = np.rint(pred[:, col]).reshape(-1, )
    class_true = eval_[:, col].reshape(-1, )

    score = accuracy_score(class_true, class_pred)
    conf = confusion_matrix(class_true, class_pred, normalize = 'true')
    pos = np.sum(class_true)
    neg = int(class_true.shape[0]) - pos
    print(conf)
    print(pos)
    print(neg)
    if pos != 0:
        tn = conf[0, 0]
        fp = conf[0, 1]
        fn = conf[1, 0]
        tp = conf[1, 1]
    else:
        tn = conf[0, 0]
        fp = np.nan
        fn = np.nan
        tp = np.nan

    print('true negative: ', tn)
    print('true positive: ', tp)
    
    print('Score for class ' + str(col) + ': {:.4f}'.format(score))
    score_li.append(score)
    zero_counts.append(neg)
    one_counts.append(pos)
    one_percent.append('{:.4f}%'.format(pos.astype(np.float32) / (pos+neg)*100))
    true_pos.append(tp)
    false_pos.append(fp)
    true_neg.append(tn)
    false_neg.append(fn)

score_li = np.vstack(score_li)
zero_counts = np.vstack(zero_counts)
one_counts = np.vstack(one_counts)
one_percent = np.vstack(one_percent)
true_pos = np.vstack(true_pos)
false_pos = np.vstack(false_pos)
true_neg = np.vstack(true_neg)
false_neg = np.vstack(false_neg)

data = np.c_[categories.reshape(-1), score_li, zero_counts, one_counts, one_percent, true_pos, false_neg, true_neg, false_pos]
df = pd.DataFrame(data = data, columns = ['categories', 'score', 'zeros_count', 'ones_count', 'ones_perc', 'true positive', 'false positive', 'true negative', 'false negative'])
print(df.head())
df.to_csv('./0423_split0.8_shuffled_scores.csv', columns = df.columns)

