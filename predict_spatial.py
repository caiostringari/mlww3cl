import pickle
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


def dir_diff(dir1, dir2):

    diff1 = (dir1 - dir2) % 360
    diff2 = (dir2 - dir1) % 360
    res = np.ones(dir1.shape)
    for i in range(len(dir1)):
        d = min(diff1[i], diff2[i])

        if dir2[i] > dir1[i]:
            res[i] = d
        else:
            res[i] = d

    return res


def main():
    # load the model
    model = keras.models.load_model(args.best_epoch)

    # load spatial data
    ds = xr.open_dataset(args.dataset)

    # load data scalers
    with open(args.hs_scaler, "rb") as file:
        hs_scaler = pickle.load(file)
    with open(args.tp_scaler, "rb") as file:
        tp_scaler = pickle.load(file)

    # time loop
    for t, _ in enumerate(ds.time.values):
        tds = ds.isel(time=t)
        print(f"processing timestep {t+1}, {len(ds.time.values)}")

        # select variables
        hs = tds["hs"].values
        t01 = tds["t01"].values
        t02 = tds["t02"].values
        spr = tds["spr"].values
        uwnd = tds["uwnd"].values
        vwnd = tds["vwnd"].values
        wdir = tds["dir"].values
        lon = tds["longitude"].values
        lat = tds["latitude"].values

        # modify directions, compute the sin and cos so it's between -1 and 1
        dm_rad = np.deg2rad(wdir) - np.pi
        dm_sinx = np.sin(dm_rad)
        dm_cosx = np.cos(dm_rad)

        # build the target array
        x = np.vstack(
            [hs, t01, t02, spr, uwnd, vwnd, wdir, dm_sinx, dm_cosx, lon, lat]
        ).T

        cols = [
            "hs",
            "t01",
            "t02",
            "spr",
            "uwnd",
            "vwnd",
            "wdir",
            "sin_dir",
            "cos_dir",
            "lon",
            "lat",
        ]
        df = pd.DataFrame(x, columns=cols)
        df = df.fillna(value=0)

        # scale
        # shape = X.shape
        cols2 = ["hs", "t01", "t02", "spr", "uwnd", "vwnd", "sin_dir", "cos_dir"]
        xscaler = MinMaxScaler(feature_range=(-1, 1)).fit(df[cols2].values)
        xp = xscaler.transform(df[cols2].values)

        # predict
        yhat = model.predict(xp)

        # scale back
        # convert sin,cos back to radians and degrees
        y_pred_rescaled = np.arctan2(yhat[:, 0], yhat[:, 1])
        dm_pred_rescaled = np.rad2deg(y_pred_rescaled + np.pi)

        # rescale Hs and TP
        hs_pred_rescaled = np.squeeze(
            hs_scaler.inverse_transform(yhat[:, 2].reshape(-1, 1))
        )
        tp_pred_rescaled = np.squeeze(
            tp_scaler.inverse_transform(yhat[:, 3].reshape(-1, 1))
        )

        # plot
        pred = np.vstack([dm_pred_rescaled, hs_pred_rescaled, tp_pred_rescaled]).T
        df_final = pd.DataFrame(pred, columns=["wdir", "hs", "tp"])
        df_final["lon"] = df["lon"].values
        df_final["lat"] = df["lat"].values

        # save the arrays
        if t == 0:
            hs_dif = np.zeros([len(ds["time"].values), len(df_final)])
            tp_dif = np.zeros([len(ds["time"].values), len(df_final)])
            dm_dif = np.zeros([len(ds["time"].values), len(df_final)])

            hs_dif[t, :] = df_final["hs"].values - df["hs"].values
            tp_dif[t, :] = df_final["tp"].values - df["t01"].values
            dm_dif[t, :] = dir_diff(df_final["wdir"].values, df["wdir"].values)

        else:
            hs_dif[t, :] = df_final["hs"].values - df["hs"].values
            tp_dif[t, :] = df_final["tp"].values - df["t01"].values
            dm_dif[t, :] = dir_diff(df_final["wdir"].values, df["wdir"].values)

    # save data for plotting
    out = {
        "hs_diff": hs_dif,
        "tp_diff": tp_dif,
        "dir_diff": dm_dif,
        "longitude": df_final["lon"].values,
        "latitude": df_final["lat"].values,
    }

    # save the scalers
    with open(args.output, "wb") as handle:
        pickle.dump(out, handle)


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument(
        "--best_epoch",
        action="store",
        dest="best_epoch",
        required=True,
        help="best epoch.",
    )

    parser.add_argument(
        "--dataset",
        action="store",
        dest="dataset",
        required=True,
        help="Input results from WW3 (.nc).",
    )

    parser.add_argument(
        "--hs_scaler",
        action="store",
        dest="hs_scaler",
        required=True,
        help="HS scaler, comes from train.py.",
    )

    parser.add_argument(
        "--tp_scaler",
        action="store",
        dest="tp_scaler",
        required=True,
        help="Tp scaler, comes from train.py.",
    )

    parser.add_argument(
        "--output",
        action="store",
        dest="output",
        required=True,
        help="Output for plottting.",
    )

    args = parser.parse_args()

    main()
