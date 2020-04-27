#!/usr/bin/env python

import numpy as np
import tensorflow
from get_12ECG_features import get_12ECG_features

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    leads, features=np.asarray(get_12ECG_features(data,header_data))
    leads = leads.reshape(1, leads.shape[0], leads.shape[1])
    
    features = features.reshape(1, features.shape[0])
    score = model.predict([leads, features])

    current_label = np.rint(score[0])
    for ix, each in enumerate(score[0]):
        current_score[ix] = each

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    filename='baseline.h5'
    loaded_model = tensorflow.keras.models.load_model(filename, compile = False)
    loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    
    return loaded_model
