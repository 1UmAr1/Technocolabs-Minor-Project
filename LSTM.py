import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Flatten, LSTM, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
import time
from keras.utils import np_utils
from sklearn.model_selection import KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
Name = "summm{}".format(int(time.time()))
tf.keras.backend.set_floatx('float64')
from sklearn.feature_extraction.text import TfidfVectorizer

Data = pd.read_csv("First_processed.csv")
X = Data["message"]
Y = Data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1103)


tfidf = TfidfVectorizer(use_idf=True, min_df=5)
tfidf.fit_transform(X_train)
tfidf.fit_transform(X_test)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
nb_classes = 4
print(X_train.shape)

X_train = tfidf.transform(X_train).toarray()
X_train = X_train[:, :, None]
print("1 DONE")


def baseline_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(254, return_sequences=True, stateful=False, input_shape=(X_train.shape[1:])))

    model.add(Bidirectional(LSTM(228, return_sequences=True, activation='tanh')))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(228, return_sequences=True, activation='tanh')))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(228, return_sequences=True, activation='tanh')))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(128, activation='relu')))
    model.add(Dense(nb_classes, activation='softmax'))
    opt = tf.keras.optimizers.Nadam(learning_rate=0.001, decay=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.summary()
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=100, verbose=True)
#kfold = KFold(n_splits=2)
#results = cross_val_score(estimator, X_train, y_train, cv=kfold, n_jobs=6,
#                          fit_params={'callbacks': [TensorBoard(log_dir="FIRST_LSTM/".format(Name))]})
estimator.fit(X_train, y_train, verbose=True)
