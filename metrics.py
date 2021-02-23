"""
Sumarize the metrics.

Use metrics.py --help for usage.
"""
import argparse

import pandas as pd

import numpy as np

from scipy.stats import pearsonr
from statsmodels.tools.eval_measures import bias
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def HH(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    HH = np.sqrt(np.sum((y_true-y_pred)**2) / np.sum(y_true*y_pred))
    return HH



def process(df, subset, variable):
    "Print the metrics on the screen."
    df1 = df.loc[df["subset"] == subset]

    rmse_ml = np.sqrt(mse(df1["{}_buoy".format(variable)],
                          df1["{}_prediction".format(variable)]))
    rmse_ww = np.sqrt(mse(df1["{}_buoy".format(variable)],
                          df1["{}_wavewatch".format(variable)]))

    mae_ml = mae(df1["{}_buoy".format(variable)],
                 df1["{}_prediction".format(variable)])
    mae_ww = mae(df1["{}_buoy".format(variable)],
                 df1["{}_wavewatch".format(variable)])

    mape_ml = mape(df1["{}_buoy".format(variable)],
                   df1["{}_prediction".format(variable)])
    mape_ww = mape(df1["{}_buoy".format(variable)],
                   df1["{}_wavewatch".format(variable)])

    bias_ml = bias(df1["{}_buoy".format(variable)],
                   df1["{}_prediction".format(variable)])
    bias_ww = bias(df1["{}_buoy".format(variable)],
                   df1["{}_wavewatch".format(variable)])

    r_ml, _ = pearsonr(df1["{}_buoy".format(variable)],
                       df1["{}_prediction".format(variable)])
    r_ww, _ = pearsonr(df1["{}_buoy".format(variable)],
                       df1["{}_wavewatch".format(variable)])

    hh_ml = HH(df1["{}_buoy".format(variable)],
               df1["{}_prediction".format(variable)])
    hh_ww = HH(df1["{}_buoy".format(variable)],
               df1["{}_wavewatch".format(variable)])

    X = np.vstack([rmse_ww, rmse_ml,
                   mae_ww, mae_ml,
                   mape_ww, mape_ml,
                   bias_ww, bias_ml,
                   r_ww, r_ml,
                   hh_ww, hh_ml]).T
    cols = ["RMSE_WW3", "RMSE_ML",
            "MAE_WW3", "MAE_ML",
            "MAPE_WW3", "MAPE_ML",
            "BIAS_WW3", "BIAS_ML",
            "R_WW3", "R_ML",
            "HH_WW3", "HH_ML"]
    table = pd.DataFrame(X, columns=cols)
    table.index = [variable + "_" + subset]
    # table.index.name = subset

    return table


def main():
    """Execute the main prgram."""

    # read the inputs
    df = pd.read_csv(args.input)

    # compute metrics
    hs_train = process(df, "train", "Hs")
    tp_train = process(df, "train", "Tp")
    dp_train = process(df, "train", "Dm")

    hs_valid = process(df, "valid", "Hs")
    tp_valid = process(df, "valid", "Tp")
    dp_valid = process(df, "valid", "Dm")

    hs_test = process(df, "test", "Hs")
    tp_test = process(df, "test", "Tp")
    dp_test = process(df, "test", "Dm")

    # concat
    dfm = pd.concat([hs_train, tp_train, dp_train,
                     hs_valid, tp_valid, dp_valid,
                     hs_test, tp_test, dp_test])
    dfm.index.name = "parameter"

    dfm.to_csv(args.output, float_format=("%.2f"))

    print(dfm)


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument("--input", "-i", action="store", dest="input",
                        required=True, help="Input data (.csv).",)

    # output data
    parser.add_argument("--output", "-o", action="store", dest="output",
                        required=True, help="Output data (.csv).",)

    args = parser.parse_args()

    main()
