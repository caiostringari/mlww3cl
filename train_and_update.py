"""
Train the model.

Use train.py --help for usage.
"""
import os
import platform
import subprocess

import datetime

import argparse

import pickle

import pandas as pd

import numpy as np

from glob import glob
from os.path import isfile

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, Input
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session

# from tensorflow_addons import metrics as tfam
from tensorflow.keras import metrics as tfm


def download_data_from_web():
    """Download data from google drive."""

    links = [
        "109BSv1yRwsjCP6woiCNyYiKUlxs5zQhh",
        "10GsgYONAzEFshUks4zWXe_4d5Uz0sboX",
        "12AIQC9K9qUjJD_2BFdfmmOHXpnaRHc1F",
        "13NKOAIyJsZ0E8xAmPkwQnJMA4y1XsrG5",
        "16RjjTiWSEnv2oCeiC8SGv0BH4jPcGxqt",
        "190nPBIPqrR9lTpd0Vwl_85AmuG0shrpr",
        "1EJYjBo2Dxn-cL1Dh-5PyNRPpsBG_Yx5s",
        "1EwNoI4hlbULXk-sza6a-4JeezQ3IG4oe",
        "1FjG3oCvCBCtd_JW9FS7YyQ3gPDgMA6yO",
        "1GjPybJFR4bX9y9mXYZ60m6ha_5uy2GEU",
        "1Gtu6xTWGaQnefyjllVl4t7gYRsrI_8k6",
        "1KxnU6QXx-J9Ld-5l8rZJ6W_Z8w9VOFfc",
        "1N02Wb_EINoHYfDHdTKSkufTr5Iwz3Nkv",
        "1Q3vtWhYiJlK_C6sUna0yRsKhDPlxzedJ",
        "1QxpyjyKj1B-b6f2UgL4KGacU77NZNIJF",
        "1RtJVipfheoEDbGJleA1ZLT1YP2Hd77gN",
        "1_fZP7QKfND90ZqAt87zBOLhVfWXlZx93",
        "1bi7N0PDlbnkoOC0Q-ZzNw0EtPRYUQcrg",
        "1c12plXjphKC0C9XEUzHEwB-XcP32RvSm",
        "1eOfq1WVROUN-irN8NC6KDj_3gnPCnjWL",
        "1fUQaoNcRq-M1O-5-Vio81_LhwRuDCGQQ",
        "1krVQJre_13VXucxplzxlw0gzvD6qidXU",
        "1oQJXLAKfLwRFkuLNjvtn5buijRXINVVB",
        "1w0FLB9Q2EsUCZp9JTfONC_oQAm8Eo69D",
        "1w3nvv9iEIeBesdJU0ZtMLwzQgbtvJS9l",
        "1z0iBC6e2KowWnB0chlpP6ZBP9H7NvFOe",
    ]

    # create data folder
    subprocess.call("mkdir -p data/", shell=True)

    for l, link in enumerate(links):
        dst = f"data/wave_data_{l+1}.csv"
        if not isfile(dst):
            print(f"downloading file {l+1} of {len(links)}", end="\r")
            cmd = f"gdown {link} --output {dst}"
            subprocess.run(cmd, shell=True, capture_output=True, check=True)
        else:
            print(f"file {l+1} of {len(links)} aleady present")
    print("\ndone downloading data\n")

    files = glob("data/*.csv")
    dfs = []
    for f in files:
        df = pd.read_csv(f, delimiter=",")
        dfs.append(df)
    inp = pd.concat(dfs)

    return inp


def MLP(shape, n_layers=2, n_neurons=512, dropout=0.25, name="my_model"):
    """Simple fully-connected MLP."""

    layers = []
    # add the input layer
    layers.append(
        Dense(
            n_neurons,
            activation="relu",
            kernel_initializer="glorot_normal",
            input_shape=shape,
        )
    )
    layers.append(Dropout(dropout))

    # add the other layers
    for _ in range(n_layers - 1):
        layers.append(Dense(n_neurons, activation="relu"))
        layers.append(Dropout(dropout))

    # add the output layer
    layers.append(Dense(4, activation="tanh"))

    # build the model
    model = Sequential(layers, name=name)

    return model


def CNN(shape=(32, 24, 1), dropout=0.25, name="my_model"):
    """Simple CNN model."""

    # define model input
    visible = Input(shape=shape)
    # add vgg module
    layer = vgg_block(visible, 64, 2)
    # add vgg module
    layer = vgg_block(layer, 128, 2)
    # add vgg module
    layer = vgg_block(layer, 256, 4)

    # flatten
    flat = Flatten()(layer)
    fcn1 = Dense(1024)(flat)
    drop = Dropout(dropout)(fcn1)
    fcn2 = Dense(1024)(drop)
    drop = Dropout(dropout)(fcn2)
    fcn3 = Dense(4)(drop)

    # create model
    model = Model(inputs=visible, outputs=fcn3, name=name)

    return model


def vgg_block(layer_in, n_filters, n_conv):
    """standard VGG block."""
    # add convolutional layers
    for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3, 3), padding="same", activation="relu")(
            layer_in
        )
    # add max pooling layer
    layer_in = MaxPooling2D((2, 2), strides=(2, 2))(layer_in)
    layer_in = Dropout(0.1)(layer_in)  # DROPOUT?
    return layer_in


def summarize_predictions(
    data, model, X, ytrue, subset, custom_scaler, scalers=[], idx=np.array([None])
):
    """Apply the model and return a dataframe with predictions."""

    yhat = np.squeeze(model.predict(X))

    # convert sin,cos back to radians and degrees
    y_pred_rescaled = np.arctan2(yhat[:, 0], yhat[:, 1])
    y_true_rescaled = np.arctan2(ytrue[:, 0], ytrue[:, 1])
    dm_pred_rescaled = np.rad2deg(y_pred_rescaled + np.pi)
    dm_true_rescaled = np.rad2deg(y_true_rescaled + np.pi)

    # rescale Hs and TP
    if custom_scaler:
        hs_pred_rescaled = min_max_scaler_inverse_transform(yhat[:, 2], max=10)
        tp_pred_rescaled = min_max_scaler_inverse_transform(yhat[:, 3], max=20)
        hs_true_rescaled = min_max_scaler_inverse_transform(ytrue[:, 2], max=10)
        tp_true_rescaled = min_max_scaler_inverse_transform(ytrue[:, 3], max=20)
    else:
        hs_pred_rescaled = np.squeeze(
            scalers[0].inverse_transform(yhat[:, 2].reshape(-1, 1))
        )
        tp_pred_rescaled = np.squeeze(
            scalers[1].inverse_transform(yhat[:, 3].reshape(-1, 1))
        )
        hs_true_rescaled = np.squeeze(
            scalers[0].inverse_transform(ytrue[:, 2].reshape(-1, 1))
        )
        tp_true_rescaled = np.squeeze(
            scalers[1].inverse_transform(ytrue[:, 3].reshape(-1, 1))
        )

    # build a dataframe
    cols = [
        "Hs_buoy",
        "Hs_prediction",
        "Tp_buoy",
        "Tp_prediction",
        "Dm_buoy",
        "Dm_prediction",
    ]
    x = np.vstack(
        [
            hs_true_rescaled,
            hs_pred_rescaled,
            tp_true_rescaled,
            tp_pred_rescaled,
            dm_true_rescaled,
            dm_pred_rescaled,
        ]
    ).T
    df = pd.DataFrame(x, columns=cols)
    df["subset"] = subset

    # add extra varibles for later plots
    if idx[0]:
        df.index = data.iloc[idx].index.values
        df.index.name = "time"
        df["location"] = data.iloc[idx]["ID"].values
        df["Hs_wavewatch"] = data.iloc[idx]["Hm0_mod"]
        df["Tp_wavewatch"] = data.iloc[idx]["Tm1_mod"]
        df["Dm_wavewatch"] = data.iloc[idx]["Dm_mod"]
    else:
        df.index = data.iloc[:].index.values
        df.index.name = "time"
        df["location"] = data.iloc[:]["ID"].values
        df["Hs_wavewatch"] = data.iloc[:]["Hm0_mod"]
        df["Tp_wavewatch"] = data.iloc[:]["Tm1_mod"]
        df["Dm_wavewatch"] = data.iloc[:]["Dm_mod"]

    return df


# Custom data scalers
def min_max_scaler_transform(x, min=0, max=20):
    """Scale x between -1 and 1."""
    y = (2 * ((x - min) / (max - x.min()))) - 1
    return y


def min_max_scaler_inverse_transform(x, min=0, max=20):
    """Reserve the scaler."""
    y = ((max - min) * ((x + 1) / 2)) + min
    return y


def main():
    """Call the main method."""

    # --- Tensorboard ---

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if platform.system().lower() == "windows":
        logdir = logs + "\\" + model_name + "\\" + date
    else:
        logdir = logs + "/" + model_name + "/" + date
    if not os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)

    tensorboard = callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1, profile_batch=1
    )

    # --- Outputs ---

    # doing it this way because TF does not pathlib it seems
    if platform.system().lower() == "windows":
        checkpoint_path = logdir + "\\" + "best_epoch.h5"
        last_epoch = logdir + "\\" + "last_epoch.h5"
        history_path = logdir + "\\" + "history.csv"
        data_out = logdir + "\\" + "predictions.csv"
        X_scaler_out = logdir + "\\" + "hs_scaler.pkl"
        hs_scaler_out = logdir + "\\" + "hs_scaler.pkl"
        tp_scaler_out = logdir + "\\" + "tp_scaler.pkl"
    else:
        checkpoint_path = logdir + "/" + "best_epoch.h5"
        last_epoch = logdir + "/" + "last_epoch.h5"
        history_path = logdir + "/" + "history.csv"
        data_out = logdir + "/" + "predictions.csv"
        X_scaler_out = logdir + "" + "hs_scaler.pkl"
        hs_scaler_out = logdir + "/" + "hs_scaler.pkl"
        tp_scaler_out = logdir + "/" + "tp_scaler.pkl"

    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )

    # --- Pre-process data ---
    inp = download_data_from_web()
    # remove one location
    print(f"locations are {inp.ID.unique()}, {len(inp.ID.unique())}")
    print("removing ALebu")
    df_alebu = inp.loc[inp["ID"] == "ALebu"]
    inp = inp.loc[inp["ID"] != "ALebu"]
    print(f"locations are {inp.ID.unique()}, {len(inp.ID.unique())}")

    df = inp.iloc[:, -25:]  # these are the scalar variables

    # get the X data based on model type
    if model_type.lower() == "mlp_parametric":
        train_features = [
            "Hm0_mod",
            "Tm1_mod",
            "Tm2_mod",
            "Spr_mod",
            "U_mod",
            "V_mod",
            "Dm_mod_sinx",
            "Dm_mod_cosx",
        ]
    else:
        ds = inp.iloc[:, 1:-25]  # this is the spectral training data

    # convert Dm_obs and Dm_mod to radians so that it is a continous
    # variable between -pi and pi

    df["Dm_mod_rad"] = np.deg2rad(df["Dm_mod"]) - np.pi
    df["Dm_obs_rad"] = np.deg2rad(df["Dm_obs"]) - np.pi

    # compute the sin and cos so it's between -1 and 1
    df["Dm_mod_sinx"] = np.sin(df["Dm_mod_rad"])
    df["Dm_mod_cosx"] = np.cos(df["Dm_mod_rad"])
    df["Dm_obs_sinx"] = np.sin(df["Dm_obs_rad"])
    df["Dm_obs_cosx"] = np.cos(df["Dm_obs_rad"])

    # prepare the training, testing and validation datasets
    if model_type == "mlp_parametric":
        X = df[train_features].values
    else:
        X = ds.values

    # prepare output array - Buoy data
    if custom_scaler:
        hs_scaled = min_max_scaler_transform(df["Hm0_obs"].values, max=10)
        tp_scaled = min_max_scaler_transform(df["Tm1_obs"].values, max=20)
    else:
        hs_scaler = MinMaxScaler(feature_range=(-1, 1))
        tp_scaler = MinMaxScaler(feature_range=(-1, 1))
        hs_scaled = np.squeeze(
            hs_scaler.fit_transform(df["Hm0_obs"].values.reshape(-1, 1))
        )
        tp_scaled = np.squeeze(
            tp_scaler.fit_transform(df["Tm1_obs"].values.reshape(-1, 1))
        )

    # target data that we are trying to predict
    y = np.vstack(
        [df["Dm_obs_sinx"].values, df["Dm_obs_cosx"].values, hs_scaled, tp_scaled]
    ).T

    # Use the location names to split the dataset according to their
    # distribution.This avoid one location being left out the training
    if stratify:
        df["labels"] = pd.factorize(df["ID"])[0].astype(np.uint16)
        labels = df["labels"].values
    else:
        labels = np.ones(y.shape)

    # get dates
    indexes = np.arange(0, len(df), 1)

    # split train/test
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indexes,
        test_size=test_size,
        stratify=labels,
        random_state=random_seed,
        shuffle=None,
    )

    # split train/val
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train,
        y_train,
        idx_train,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )

    print(
        "\nThere are {} training samples, "
        "{} validation samples, and {} testing samples".format(
            X_train.shape[0], X_val.shape[0], X_test.shape[0]
        )
    )

    # scale data -1, -1 interval
    if custom_scaler:
        shape = X_train.shape
        X_train = min_max_scaler_transform(X_train.flatten(), min=0, max=5)
        X_train = X_train.reshape(shape)

        shape = X_test.shape
        X_test = min_max_scaler_transform(X_test.flatten(), min=0, max=5)
        X_test = X_test.reshape(shape)

        shape = X_val.shape
        X_val = min_max_scaler_transform(X_val.flatten(), min=0, max=5)
        X_val = X_val.reshape(shape)
    else:
        xscaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
        X_train = xscaler.transform(X_train)
        X_test = xscaler.transform(X_test)
        X_val = xscaler.transform(X_val)

    """### Trainining"""

    # --- Model ---

    # start by clearing the section
    clear_session()

    # reshape the data. This a very important step.
    # we need to reshape the data so that it is the the shape
    # that tensorflow likes.

    if model_type.lower() == "cnn_spectral":

        # reshape for the CNN
        X_train = X_train.reshape(X_train.shape[0], spc_size[0], spc_size[1], 1)
        X_test = X_test.reshape(X_test.shape[0], spc_size[0], spc_size[1], 1)
        X_val = X_val.reshape(X_val.shape[0], spc_size[0], spc_size[1], 1)

        # define the model
        model = CNN(
            shape=(spc_size[0], spc_size[1], 1), dropout=dropout, name=model_name
        )

    else:
        model = MLP([X_train.shape[1]], nlayers, nneurons, dropout, name=model_name)

    # compile
    metrics = [
        tfm.RootMeanSquaredError(name="RMSE"),
        tfm.MeanSquaredError(name="MSE"),
        tfm.MeanAbsoluteError(name="MAE"),
        tfm.MeanAbsolutePercentageError(name="MAPE"),
        tfm.MeanSquaredLogarithmicError(name="MSLE"),
    ]
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=metrics)
    model.summary()

    # train
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard, checkpoint],
    )
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    hist.to_csv(history_path)

    # save last epoch
    model.save(last_epoch)

    # load the best epoch from file
    best = load_model(checkpoint_path)

    # predict for all subsets
    if custom_scaler:
        # train
        df_train = summarize_predictions(
            df, best, X_train, y_train, "train", True, idx=idx_train
        )
        # valid
        df_valid = summarize_predictions(
            df, best, X_val, y_val, "valid", True, idx=idx_val
        )
        # test
        df_test = summarize_predictions(
            df, best, X_test, y_test, "test", True, idx=idx_test
        )
    else:
        scalers = [hs_scaler, tp_scaler]
        # train
        df_train = summarize_predictions(
            df, best, X_train, y_train, "train", False, scalers=scalers, idx=idx_train
        )
        # valid
        df_valid = summarize_predictions(
            df, best, X_val, y_val, "valid", False, scalers=scalers, idx=idx_val
        )
        # test
        df_test = summarize_predictions(
            df, best, X_test, y_test, "test", False, scalers=scalers, idx=idx_test
        )

    # save the scalers
    with open(X_scaler_out, "wb") as handle:
        pickle.dump(xscaler, handle)
    with open(hs_scaler_out, "wb") as handle:
        pickle.dump(hs_scaler, handle)
    with open(tp_scaler_out, "wb") as handle:
        pickle.dump(tp_scaler, handle)

    # save the metrics
    dfo = pd.concat([df_train, df_valid, df_test])
    dfo.to_csv(data_out)

    print("\n\n--- Updating the model -- \n\n")

    df_up = df_alebu  # new data

    # prepare the training, testing and validation datasets
    if model_type == "mlp_parametric":
        X_up = df_up[train_features].values
    else:
        X_up = df_up.iloc[:, 1:-25].values

    # convert Dm_obs and Dm_mod to radians so that it is a continous
    # variable between -pi and pi

    df_up["Dm_mod_rad"] = np.deg2rad(df_up["Dm_mod"]) - np.pi
    df_up["Dm_obs_rad"] = np.deg2rad(df_up["Dm_obs"]) - np.pi

    # compute the sin and cos so it's between -1 and 1
    df_up["Dm_mod_sinx"] = np.sin(df_up["Dm_mod_rad"])
    df_up["Dm_mod_cosx"] = np.cos(df_up["Dm_mod_rad"])
    df_up["Dm_obs_sinx"] = np.sin(df_up["Dm_obs_rad"])
    df_up["Dm_obs_cosx"] = np.cos(df_up["Dm_obs_rad"])

    # scale X
    shape = X_up.shape
    X_up = min_max_scaler_transform(X_up.flatten(), min=0, max=5)
    X_up = X_up.reshape(shape)

    # prepare output array - Buoy data
    hs_scaled = min_max_scaler_transform(df_up["Hm0_obs"].values, max=10)
    tp_scaled = min_max_scaler_transform(df_up["Tm1_obs"].values, max=20)

    # target data that we are trying to predict
    y_up = np.vstack([df_up["Dm_obs_sinx"].values,
                    df_up["Dm_obs_cosx"].values,
                    hs_scaled,
                    tp_scaled]).T
    # get dates
    indexes = np.arange(0, len(y_up), 1)

    # split train/test
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_up, y_up, indexes, test_size=0.25, stratify=None,
        random_state=random_seed, shuffle=True)

    # split train/val
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train, y_train, idx_train, test_size=0.25,
        random_state=random_seed, shuffle=True)

    print("\nThere are {} training samples, "
            "{} validation samples, and {} testing samples".format(
                X_train.shape[0], X_val.shape[0], X_test.shape[0]))

    # scale data -1 to 1 interval
    shape = X_train.shape
    X_train = min_max_scaler_transform(X_train.flatten(), min=0, max=5)
    X_train = X_train.reshape(shape)

    shape = X_test.shape
    X_test = min_max_scaler_transform(X_test.flatten(), min=0, max=5)
    X_test = X_test.reshape(shape)

    shape = X_val.shape
    X_val = min_max_scaler_transform(X_val.flatten(), min=0, max=5)
    X_val = X_val.reshape(shape)

    # start by clearing the section
    clear_session()

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    up_logdir = logs + "/" + "alebu" + "/" + date
    tensorboard = callbacks.TensorBoard(log_dir=up_logdir,
                                        histogram_freq=1,
                                        profile_batch=1)
    up_checkpoint_path = up_logdir + "/" + "best_epoch.h5"
    up_checkpoint = callbacks.ModelCheckpoint(filepath=up_checkpoint_path,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            monitor='val_loss',
                                            mode="min",
                                            verbose=1)

    # load previous best model
    model = load_model(checkpoint_path)

    # train
    if model_type != "cnn_spectral":
        shape = [X_train.shape[0], model.input.shape[1]]
    else:
        shape = [X_train.shape[0], model.input.shape[1], model.input.shape[2], 1]
    X_train = X_train.reshape(shape)

    if model_type != "mlp_parametric":
        shape = [X_val.shape[0], model.input.shape[1]]
    else:
        shape = [X_val.shape[0], model.input.shape[1], model.input.shape[2], 1]

    X_val = X_val.reshape(shape)

    up_history = model.fit(X_train, y_train,
                        batch_size=X_train.shape[0],
                        epochs=2,
                        verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=[tensorboard, up_checkpoint])
    hist = pd.DataFrame(up_history.history)
    hist["epoch"] = up_history.epoch
    hist.to_csv(up_logdir + "/alebu_history.csv")




if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument(
        "--model",
        "-m",
        action="store",
        dest="model",
        required=True,
        help="Model name.",
    )

    parser.add_argument(
        "--type",
        "-t",
        action="store",
        dest="model_type",
        required=True,
        help="Model type. "
        "Possible choices are: cnn_spectral, mlp_parametric "
        " or mlp_spectral.",
    )

    # loggir
    parser.add_argument(
        "--logdir",
        "-logdir",
        action="store",
        required=True,
        dest="logdir",
        help="Logging directory for Tensorboard.",
    )

    # random state seed for reproducibility
    parser.add_argument(
        "--random-state",
        "-random-state",
        action="store",
        dest="random_state",
        default=11,
        required=False,
        help="Random state.",
    )
    # validation size
    parser.add_argument(
        "--test-size",
        "-testsize",
        action="store",
        dest="test_size",
        default=0.3,
        required=False,
        help="Test set size. Default is 0.3",
    )

    # layers
    parser.add_argument(
        "--layers",
        "-nl",
        action="store",
        dest="nlayers",
        default=3,
        required=False,
        help="Number of layers. Only for MLPs.",
    )
    # neuros
    parser.add_argument(
        "--neurons",
        "-nn",
        action="store",
        dest="nneurons",
        default=512,
        required=False,
        help="Number of neurons per layer. Only for MLPs.",
    )

    # learning rate
    parser.add_argument(
        "--learning-rate",
        "-lr",
        action="store",
        dest="learning_rate",
        default=10e-6,
        required=False,
        help="Learning rate. Default is 10E-6.",
    )
    # dropout
    parser.add_argument(
        "--dropout",
        "-dropout",
        action="store",
        dest="dropout",
        default=0.25,
        required=False,
        help="Dropout rate. Default is 0.25.",
    )

    # epochs
    parser.add_argument(
        "--epochs",
        "-epochs",
        action="store",
        dest="epochs",
        default=256,
        required=False,
        help="Number of epochs. Default is 256.",
    )
    # batch size
    parser.add_argument(
        "--batch-size",
        "-batch-size",
        action="store",
        dest="batch_size",
        default=2048,
        required=False,
        help="Batch size. Default is 2048.",
    )
    # stratify
    parser.add_argument(
        "--stratify",
        "-stratify",
        action="store",
        dest="stratify",
        default=True,
        required=False,
        help="Use class stratification (by location).",
    )

    # input size
    parser.add_argument(
        "--input-size",
        "-input-size",
        nargs=2,
        action="store",
        dest="size",
        default=[32, 24],
        required=False,
        help="2d-spectrum size. Default is 32x24.",
    )

    # input size
    parser.add_argument(
        "--custom_scaler",
        "-custom_scaler",
        action="store_true",
        dest="custom_scaler",
        required=False,
        help="Use custom data scaler.",
    )

    args = parser.parse_args()

    # --- constants ---
    model_name = args.model
    spc_size = (int(args.size[0]), int(args.size[1]))
    batch_size = int(args.batch_size)
    random_seed = int(args.random_state)
    epochs = int(args.epochs)
    test_size = float(args.test_size)
    learning_rate = float(args.learning_rate)
    stratify = bool(args.stratify)
    dropout = int(args.dropout)
    logs = args.logdir
    nlayers = int(args.nlayers)
    nneurons = int(args.nneurons)
    model_type = args.model_type
    if model_type not in ["cnn_spectral", "mlp_parametric", "mlp_spectral"]:
        raise IOError(
            ("Model type must be cnn_spectral or mlp_parametric " "or mlp_spectral")
        )
    if args.custom_scaler:
        custom_scaler = True
    else:
        custom_scaler = False

    # call the main program
    main()

    print("\nMy work is done!")
