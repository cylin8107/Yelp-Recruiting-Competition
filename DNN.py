#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Mining final project: DNN

@author: Jui-Hsiu, Hsu
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_log_error

from keras import backend as K
import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.layers.normalization import BatchNormalization

# 1. Read Data
test_dir = "data/yelp_test_set/"
test_set_review = pd.read_json(os.path.join(test_dir, 'yelp_test_set_review.json'), lines=True)

# 2. Read Original Features
X = pd.read_pickle("feature/X.pkl")
Y = pd.read_pickle("feature/Y.pkl")
test_X = pd.read_pickle("feature/test_X.pkl")

# 3. Features Engineering
del X['business_open']
del X['business_stars']
del X['user_average_stars']
del test_X['business_open']
del test_X['business_stars']
del test_X['user_average_stars']
reverse_X = 1/X
reverse_test_X = 1/test_X
reverse_X.replace(to_replace=np.inf, value=0, inplace=True)
reverse_test_X.replace(to_replace=np.inf, value=0, inplace=True)

poly = PolynomialFeatures(degree=2, include_bias=False)
X = pd.DataFrame(poly.fit_transform(X))
test_X = pd.DataFrame(poly.fit_transform(test_X))

X = np.hstack([X, np.array(reverse_X)])
test_X = np.hstack([test_X, np.array(reverse_test_X)])

# 4. Check Correlation
corr = pd.DataFrame(np.hstack([X, Y])).corr()
#corr = pd.concat([X, Y], axis=1).corr()

# 5. Standardization
scaler = StandardScaler().fit(X)
stand_X = scaler.transform(X, copy=True)
stand_test_X = scaler.transform(test_X, copy=True)

# 6. DNN Model
def rmsle_K(y, y0):
    return K.sqrt(K.mean(K.square(tf.log1p(y) - tf.log1p(y0))))
'''
DNN_input = Input(shape=(stand_X.shape[1],))
DNN_layer1 = Dense(units=64)(DNN_input)
DNN_batch1 = BatchNormalization()(DNN_layer1)
DNN_layer2 = Dense(units=32)(DNN_batch1)
DNN_batch2 = BatchNormalization()(DNN_layer2)
DNN_output = Dense(units=1, activation="relu")(DNN_batch2)
model = Model(inputs=DNN_input, outputs=DNN_output)

model.compile(loss=rmsle_K, optimizer='adam', metrics=["mse"])

batch_size = 256
n_epoch = 10

history = model.fit(stand_X, Y, batch_size=batch_size, epochs=n_epoch)
model.save("model/DNN.h5")
'''
model = load_model("model/DNN.h5", custom_objects={'rmsle_K': rmsle_K})

pred_Y = model.predict(stand_X)
pred_Y[pred_Y<0] = 0
print("RMSLE score = ", np.sqrt(mean_squared_log_error(Y, pred_Y)))

# 7. Submit
prediction = model.predict(stand_test_X)
prediction[prediction<0] = 0
prediction = pd.DataFrame(prediction, columns=['Votes'])
prediction.insert(0, 'Id', test_set_review['review_id'])
prediction.to_csv('submit/DNN.csv', index=False)