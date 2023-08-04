"""
This is a basic example of what you need to do to participate to the tournament.
The code will not have access to the internet (or any socket related operation).
"""

# Imports
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import load_model
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import pickle
import math


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def convert_to_pairwise_train(X_train, y_train):
    pairs = []
    labels = []
    ids = []
    n_samples = X_train.shape[0]
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            pairs.append([X_train[i, 2:], X_train[j, 2:]])
            ids.append([X_train[i, :2], X_train[j, :2]])
            labels.append(1 if y_train[i] > y_train[j] else 0)
    return np.array(pairs).astype('float32'), np.array(labels).astype('float32'), np.array(ids)

def convert_to_pairwise_test(X_test):
    pairs = []
    ids = []
    n_samples = X_test.shape[0]
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            pairs.append([X_test[i, 2:], X_test[j, 2:]])
            ids.append([X_test[i, :2], X_test[j, :2]])
    return np.array(pairs).astype('float32'), np.array(ids)

def train(X_train: pd.DataFrame, y_train: pd.DataFrame, model_directory_path: str = "resources") -> None:
    # X_train_dates = list(set(X_train['date']))[-40:]
    # X_train_orig = X_train[X_train['date'].isin(X_train_dates)].copy()
    # y_train_orig = y_train[X_train['date'].isin(X_train_dates)].copy()
    #
    # X_train = X_train[X_train['date'].isin(X_train_dates)]
    # y_train = X_train[X_train['date'].isin(X_train_dates)]
    #
    # #Scaling
    # scaler = StandardScaler()
    # X_ids = np.asarray(X_train[['date', 'id']])
    # X_scale_pca = X_train.drop(columns=['date', 'id'])
    # X_scale_pca = scaler.fit_transform(X_scale_pca)
    #
    # #PCA
    # n_components = 40
    # pca = PCA(n_components=n_components)
    # pca_features = pca.fit_transform(X_scale_pca)
    # X_train_concat = np.concatenate((X_ids, pca_features), axis=1)
    # y_train = np.asarray(y_train)
    #
    # #Save out Scaler and PCA
    # with open(Path(model_directory_path) / 'scaler.pkl', 'wb') as file:
    #     pickle.dump(scaler, file)
    #
    # with open(Path(model_directory_path) / 'pca.pkl', 'wb') as file:
    #     pickle.dump(pca, file)
    #
    # #Begin Dates Processing
    # date_list = list(set(X_train_concat[:,0]))
    # dates_array = list(split(date_list, 20))
    #
    #
    # for dates_list in dates_array:
    #     print(dates_list)
    #     X_train_pairs = np.empty((0, 2, 40))
    #     y_train_labels = np.empty((0,))
    #     X_train_ids = np.empty((0, 2, 2))
    #
    #     for date in dates_list:
    #         X_for_pairs = X_train_concat[X_train_concat[:,0] == date]
    #         y_for_pairs = y_train[y_train[:,0] == date][:,2]
    #         X_train_pair_array, y_train_labels_array, X_train_ids_array = convert_to_pairwise_train(X_for_pairs, y_for_pairs)
    #
    #         X_train_pairs = np.concatenate((X_train_pairs, X_train_pair_array), axis=0)
    #         y_train_labels = np.concatenate((y_train_labels, y_train_labels_array), axis=0)
    #         X_train_ids = np.concatenate((X_train_ids, X_train_ids_array), axis=0)
    #
    #
    #
    #     #Train Test Split
    #     X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_train_pairs, y_train_labels, random_state=42, shuffle=True, test_size=0.3)
    #     del X_train_pairs
    #     del y_train_labels
    #
    #
    #
    #     #Model Training
    #     model_pathname = Path('resources') / "model.keras"
    #
    #     if model_pathname.is_file():
    #         print(f"Opened Model for Date {date}")
    #
    #         model = load_model(model_pathname)
    #
    #         history = model.fit(
    #             X_train_nn,
    #             y_train_nn,
    #             batch_size=5000,
    #             epochs=10,
    #             validation_data=[X_test_nn, y_test_nn],
    #             callbacks=[mc, early_stopping],
    #             shuffle=False,
    #             use_multiprocessing=True,
    #             verbose=0
    #         )
    #
    #     else:
    #         #Neural Network Model
    #         mc = ModelCheckpoint(model_pathname, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    #
    #         early_stopping = EarlyStopping(
    #             monitor='val_loss',
    #             patience=1,
    #             verbose=0,
    #             mode='auto',
    #             baseline=None,
    #             restore_best_weights=True)
    #
    #         model = keras.Sequential([
    #             keras.layers.Dense(800, activation='relu', kernel_initializer='lecun_normal', input_shape=(X_train_nn.shape[1], X_train_nn.shape[2])),
    #             keras.layers.BatchNormalization(),
    #             keras.layers.Dense(500, activation='relu', kernel_initializer='lecun_normal'),
    #             keras.layers.BatchNormalization(),
    #             keras.layers.Dense(250, activation='relu', kernel_initializer='lecun_normal'),
    #             keras.layers.BatchNormalization(),
    #             keras.layers.Dense(100, activation='relu', kernel_initializer='lecun_normal'),
    #             keras.layers.BatchNormalization(),
    #             keras.layers.Flatten(),
    #             keras.layers.Dense(1, activation='sigmoid', kernel_initializer='lecun_normal')
    #         ])
    #
    #         optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    #
    #         model.compile(optimizer=optimizer,
    #                       loss='binary_crossentropy',
    #                       metrics=['accuracy'])
    #
    #         history = model.fit(
    #             X_train_nn,
    #             y_train_nn,
    #             batch_size=10000,
    #             epochs=10,
    #             validation_data=[X_test_nn, y_test_nn],
    #             callbacks=[mc, early_stopping],
    #             shuffle=True,
    #             use_multiprocessing=True,
    #             verbose=0
    #         )
    #
    #         model.save(model_pathname)
    #
    #     print(f"Finished training for Date {dates_list}")
    # print("Finished All Training")

    print('Finished training')

    # make sure that the train function correctly save the trained model
    # in the model_directory_path
    # print(f"Saving model in {model_pathname}")
    # joblib.dump(model, model_pathname)


def infer(X_test: pd.DataFrame, model_directory_path: str = "resources") -> pd.DataFrame:
    X_test_orig = X_test.copy()
    dates = list(X_test_orig['date'].unique())

    #Load Scaler
    with open(Path(model_directory_path) / 'scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    #Load PCA
    with open(Path(model_directory_path) / 'pca.pkl', 'rb') as file:
        pca = pickle.load(file)

    #Scaling
    X_ids = np.asarray(X_test[['date', 'id']])
    X_scale_pca = X_test.drop(columns=['date', 'id'])
    X_scale_pca = scaler.transform(X_scale_pca)

    #PCA
    pca_features = pca.transform(X_scale_pca)
    X_test_concat = np.concatenate((X_ids, pca_features), axis=1)

    result_df = pd.DataFrame(columns=['date', 'id', 'value'])

    for date in dates:
        X_test_date = X_test_orig[X_test_orig['date'] == date]
        X_for_pairs = X_test_concat[X_test_concat[:,0] == date]

        #Load Model
        model_pathname = Path(model_directory_path) / "model.keras"
        model = load_model(model_pathname)

        #Pairwise Transformation
        X_test_pairs, X_test_ids = convert_to_pairwise_test(X_for_pairs)

        print(f"Predicting for Date {date} in Test")
        preds = model.predict(X_test_pairs, batch_size=3000)

        preds_df_1 = pd.DataFrame({'id': X_test_ids[:,0,1].flatten(), 'date': X_test_ids[:,0,0].flatten(), 'value': preds.flatten()})

        result = preds_df_1.groupby(['date', 'id']).mean().reset_index()

        result = pd.merge(X_test_date, result, on=['id', 'date'], how='left')

        result = result[['date', 'id', 'value']]

        result['value'] = result['value'].fillna(result['value'].mean())

        minmax = MinMaxScaler(feature_range=(-1, 1))

        # Scale the 'Values' column
        result['value'] = minmax.fit_transform(result[['value']])

        result_df = pd.concat([result_df, result], ignore_index=False, axis=0)

        print(f"Finished predictions for Date {date} in Test")
    print("Finished All Predictions")

    return result_df
