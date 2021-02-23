"""
Plot bar plots with the metrics.

Use bars.py --help for usage.
"""
import argparse

import pandas as pd
import numpy as np

from scipy.stats import pearsonr

import colorcet as cc

from string import ascii_lowercase

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sns.set_context("paper", font_scale=1.35, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {"axes.linewidth": 2,
                        "legend.frameon": True,
                        "axes.facecolor": "w",
                        "grid.color": "w"})
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams['patch.edgecolor'] = "k"


def main():
    pass



if __name__ == '__main__':


    # Argument parser
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument("--input", "-i", nargs=3, action="store", dest="input",
                        required=True,
                        help="Input data MPL_PAR, MPL_SPC, CNN_SCP (.csv).",)

    # output figure
    parser.add_argument("--output", "-o", action="store", dest="output",
                        required=True, help="Output figure name (.png).",)

    args = parser.parse_args()

    df_par = pd.read_csv(args.input[0], index_col="parameter")
    df_spc = pd.read_csv(args.input[1], index_col="parameter")
    df_cnn = pd.read_csv(args.input[2], index_col="parameter")

    # open a new figure
    fig = plt.figure(figsize=(9, 12), constrained_layout=False)
    spec = gridspec.GridSpec(ncols=3, nrows=4, figure=fig)

    ax01 = fig.add_subplot(spec[0, 0])
    ax02 = fig.add_subplot(spec[0, 1])
    ax03 = fig.add_subplot(spec[0, 2])

    ax04 = fig.add_subplot(spec[1, 0])
    ax05 = fig.add_subplot(spec[1, 1])
    ax06 = fig.add_subplot(spec[1, 2])

    ax07 = fig.add_subplot(spec[2, 0])
    ax08 = fig.add_subplot(spec[2, 1])
    ax09 = fig.add_subplot(spec[2, 2])

    ax10 = fig.add_subplot(spec[3, 0])
    ax11 = fig.add_subplot(spec[3, 1])
    ax12 = fig.add_subplot(spec[3, 2])

    axes = [ax01, ax02, ax03, ax04, ax05, ax06,
            ax07, ax08, ax09, ax10, ax11, ax12]
    x = [0, 1, 2, 3]
    labels = [r"$WW3$", r"$MPL_{PAR}$", r"$MPL_{SPC}$", r"$CNN_{SPC}$"]
    colors = sns.color_palette("colorblind").as_hex()

    # RMSE
    ax01.bar(0, df_par["RMSE_WW3"].T["Hs_test"], color=colors[0])
    ax01.bar(1, df_par["RMSE_ML"].T["Hs_test"], color=colors[1])
    ax01.bar(2, df_spc["RMSE_ML"].T["Hs_test"], color=colors[2])
    ax01.bar(3, df_spc["RMSE_ML"].T["Hs_test"], color=colors[3])
    ax01.set_ylabel(r"$H_{m0}$ $RMSE$ $[m]$")

    ax02.bar(0, df_par["RMSE_WW3"].T["Tp_test"], color=colors[0])
    ax02.bar(1, df_par["RMSE_ML"].T["Tp_test"], color=colors[1])
    ax02.bar(2, df_spc["RMSE_ML"].T["Tp_test"], color=colors[2])
    ax02.bar(3, df_spc["RMSE_ML"].T["Tp_test"], color=colors[3])
    ax02.set_ylabel(r"$T_{m01}$ $RMSE$ $[s]$")

    ax03.bar(0, df_par["RMSE_WW3"].T["Dm_test"], color=colors[0])
    ax03.bar(1, df_par["RMSE_ML"].T["Dm_test"], color=colors[1])
    ax03.bar(2, df_spc["RMSE_ML"].T["Dm_test"], color=colors[2])
    ax03.bar(3, df_spc["RMSE_ML"].T["Dm_test"], color=colors[3])
    ax03.set_ylabel(r"$D_{m}$ $RMSE$ $[^{o}]$")

    # MAPE
    ax04.bar(0, df_par["MAPE_WW3"].T["Hs_test"], color=colors[0])
    ax04.bar(1, df_par["MAPE_ML"].T["Hs_test"], color=colors[1])
    ax04.bar(2, df_spc["MAPE_ML"].T["Hs_test"], color=colors[2])
    ax04.bar(3, df_spc["MAPE_ML"].T["Hs_test"], color=colors[3])
    ax04.set_ylabel(r"$H_{m0}$ $MAPE$ $[\%]$")

    ax05.bar(0, df_par["MAPE_WW3"].T["Tp_test"], color=colors[0])
    ax05.bar(1, df_par["MAPE_ML"].T["Tp_test"], color=colors[1])
    ax05.bar(2, df_spc["MAPE_ML"].T["Tp_test"], color=colors[2])
    ax05.bar(3, df_spc["MAPE_ML"].T["Tp_test"], color=colors[3])
    ax05.set_ylabel(r"$T_{m01}$ $MAPE$ $[\%]$")

    ax06.bar(0, df_par["MAPE_WW3"].T["Dm_test"], color=colors[0])
    ax06.bar(1, df_par["MAPE_ML"].T["Dm_test"], color=colors[1])
    ax06.bar(2, df_spc["MAPE_ML"].T["Dm_test"], color=colors[2])
    ax06.bar(3, df_spc["MAPE_ML"].T["Dm_test"], color=colors[3])
    ax06.set_ylabel(r"$D_{m}$ $MAPE$ $[\%]$")

    # BIAS
    ax07.bar(0, df_par["BIAS_WW3"].T["Hs_test"], color=colors[0])
    ax07.bar(1, df_par["BIAS_ML"].T["Hs_test"], color=colors[1])
    ax07.bar(2, df_spc["BIAS_ML"].T["Hs_test"], color=colors[2])
    ax07.bar(3, df_spc["BIAS_ML"].T["Hs_test"], color=colors[3])
    ax07.set_ylabel(r"$H_{m0}$ $Bias$ $[m]$")

    ax08.bar(0, df_par["BIAS_WW3"].T["Tp_test"], color=colors[0])
    ax08.bar(1, df_par["BIAS_ML"].T["Tp_test"], color=colors[1])
    ax08.bar(2, df_spc["BIAS_ML"].T["Tp_test"], color=colors[2])
    ax08.bar(3, df_spc["BIAS_ML"].T["Tp_test"], color=colors[3])
    ax08.set_ylabel(r"$T_{m01}$ $Bias$ $[s]$")

    ax09.bar(0, df_par["BIAS_WW3"].T["Dm_test"], color=colors[0])
    ax09.bar(1, df_par["BIAS_ML"].T["Dm_test"], color=colors[1])
    ax09.bar(2, df_spc["BIAS_ML"].T["Dm_test"], color=colors[2])
    ax09.bar(3, df_spc["BIAS_ML"].T["Dm_test"], color=colors[3])
    ax09.set_ylabel(r"$D_{m}$ $Bias$ $[^{o}]$")

    # HH1985
    ax10.bar(0, df_par["HH_WW3"].T["Hs_test"], color=colors[0])
    ax10.bar(1, df_par["HH_ML"].T["Hs_test"], color=colors[1])
    ax10.bar(2, df_spc["HH_ML"].T["Hs_test"], color=colors[2])
    ax10.bar(3, df_spc["HH_ML"].T["Hs_test"], color=colors[3])
    ax10.set_ylabel(r"$H_{m0}$ $HH 1985$ $[m]$")

    ax11.bar(0, df_par["HH_WW3"].T["Tp_test"], color=colors[0])
    ax11.bar(1, df_par["HH_ML"].T["Tp_test"], color=colors[1])
    ax11.bar(2, df_spc["HH_ML"].T["Tp_test"], color=colors[2])
    ax11.bar(3, df_spc["HH_ML"].T["Tp_test"], color=colors[3])
    ax11.set_ylabel(r"$T_{m01}$ $HH 1985$ $[s]$")

    ax12.bar(0, df_par["HH_WW3"].T["Dm_test"], color=colors[0])
    ax12.bar(1, df_par["HH_ML"].T["Dm_test"], color=colors[1])
    ax12.bar(2, df_spc["HH_ML"].T["Dm_test"], color=colors[2])
    ax12.bar(3, df_spc["HH_ML"].T["Dm_test"], color=colors[3])
    ax12.set_ylabel(r"$D_{m}$ $HH 1985$ $[^{o}]$")

    # set axes
    for k, ax in enumerate(axes):
        ax.set_xticks(x)
        ax.set_xticklabels("")
        sns.despine(ax=ax)
        ax.set_ylim(ax.get_ylim()[0], 1.15*ax.get_ylim()[1])
        ax.text(0.05, 0.975, "(" + ascii_lowercase[k] + ")", fontsize=14,
                va="top", zorder=100, transform=ax.transAxes, ha="left",
                bbox=dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7))
    for ax in [ax10, ax11, ax12]:
        ax.set_xticklabels(labels, rotation=90)

    fig.tight_layout()
    plt.savefig(args.output, dpi=300, pad_inches=0.1,
                bbox_inches='tight')
    plt.show()
