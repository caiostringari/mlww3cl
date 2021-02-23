"""
Plot regressions.

Use regressions.py --help for usage.
"""
import argparse

import pandas as pd
import numpy as np

from scipy.stats import pearsonr

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
    """Call the main program."""
    df_par = pd.read_csv(args.input[0])
    df_spc = pd.read_csv(args.input[1])
    df_cnn = pd.read_csv(args.input[2])

    df_par = df_par.loc[df_par["subset"] == args.subset]
    df_spc = df_spc.loc[df_spc["subset"] == args.subset]
    df_cnn = df_cnn.loc[df_cnn["subset"] == args.subset]

    # open a new figure
    fig = plt.figure(figsize=(9, 12), constrained_layout=False)
    spec = gridspec.GridSpec(ncols=3, nrows=4, figure=fig)

    ax01 = fig.add_subplot(spec[0, 0])
    ax02 = fig.add_subplot(spec[1, 0])
    ax03 = fig.add_subplot(spec[2, 0])
    ax04 = fig.add_subplot(spec[3, 0])

    ax05 = fig.add_subplot(spec[0, 1])
    ax06 = fig.add_subplot(spec[1, 1])
    ax07 = fig.add_subplot(spec[2, 1])
    ax08 = fig.add_subplot(spec[3, 1])

    ax09 = fig.add_subplot(spec[0, 2])
    ax10 = fig.add_subplot(spec[1, 2])
    ax11 = fig.add_subplot(spec[2, 2])
    ax12 = fig.add_subplot(spec[3, 2])

    axes = [ax01, ax05, ax09, ax02, ax06, ax10,
            ax03, ax07, ax11, ax04, ax08, ax12]

    skws = {"facecolor": "k", "edgecolor": "0.5", "s": 30, "alpha": 0.25}
    box = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)

    # Hm0 comparison
    xs = ["Hs_buoy", "Hs_buoy", "Hs_buoy", "Hs_buoy"]
    ys = ["Hs_wavewatch", "Hs_prediction", "Hs_prediction", "Hs_prediction"]
    axs = [ax01, ax02, ax03, ax04]
    dfs = [df_par, df_par, df_spc, df_cnn]
    ylbs = ["WWIII $H_{m0}$ $[m]$", "$MLP_{PAR}$ $H_{m0}$ $[m]$",
            "$MLP_{SPC}$ $H_{m0}$ $[m]$", "$CNN_{SPC}$ $H_{m0}$ $[m]$"]
    xlabel = "Observed $H_{m0}$ $[m]$"

    for ax, x, y, df, ylabel in zip(axs, xs, ys, dfs, ylbs):
        sns.regplot(ax=ax, data=df, x=x, y=y, scatter=True, color="r",
                    scatter_kws=skws)
        # ax.hexbin(df[x].values, df[y].values, cmap=cm, bins=15)
        vmax = max(df[x].max(), df[y].max())
        vmin = min(df[x].min(), df[y].min())
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot((vmin, vmax), (vmin, vmax), color="dodgerblue", ls="--", lw=2)
        r, p = pearsonr(df[x].values, df[y].values)
        txt = "$r_{{xy}}$={0:.2f}\n$p$<0.05".format(r)
        ax.text(0.95, 0.05, txt, fontsize=10,
                va="bottom", zorder=100, transform=ax.transAxes, ha="right",
                bbox=box)

    # Tm01 comparison
    xs = ["Tp_buoy", "Tp_buoy", "Tp_buoy", "Tp_buoy"]
    ys = ["Tp_wavewatch", "Tp_prediction", "Tp_prediction", "Tp_prediction"]
    axs = [ax05, ax06, ax07, ax08]
    dfs = [df_par, df_par, df_spc, df_cnn]
    ylbs = ["WWIII $T_{m01}$ $[s]$", "$MLP_{PAR}$ $T_{m01}$ $[s]$",
            "$MLP_{SPC}$ $T_{m01}$ $[s]$", "$CNN_{SPC}$ $T_{m01}$ $[s]$"]
    xlabel = "Observed $T_{m01}$ $[s]$"

    for ax, x, y, df, ylabel in zip(axs, xs, ys, dfs, ylbs):
        sns.regplot(ax=ax, data=df, x=x, y=y, scatter=True, color="r",
                    scatter_kws=skws)
        vmax = max(df[x].max(), df[y].max())
        vmin = min(df[x].min(), df[y].min())
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot((vmin, vmax), (vmin, vmax), color="dodgerblue", ls="--", lw=2)
        r, p = pearsonr(df[x].values, df[y].values)
        txt = "$r_{{xy}}$={0:.2f}\n$p$<0.05".format(r)
        ax.text(0.95, 0.05, txt, fontsize=10,
                va="bottom", zorder=100, transform=ax.transAxes, ha="right",
                bbox=box)

    # Dm comparison
    xs = ["Dm_buoy", "Dm_buoy", "Dm_buoy", "Dm_buoy"]
    ys = ["Dm_wavewatch", "Dm_prediction", "Dm_prediction", "Dm_prediction"]
    axs = [ax09, ax10, ax11, ax12]
    dfs = [df_par, df_par, df_spc, df_cnn]
    ylbs = ["WWIII $D_m$ $[^{o}]$", "$MLP_{PAR}$ $D_m$ $[^{o}]$",
            "$MLP_{SPC}$ $D_m$ $[^{o}]$", "$CNN_{SPC}$ $D_m$ $[^{o}]$"]
    xlabel = "Observed $D_{m}$  $[^{o}]$"

    for ax, x, y, df, ylabel in zip(axs, xs, ys, dfs, ylbs):
        sns.regplot(ax=ax, data=df, x=x, y=y, scatter=True, color="r",
                    scatter_kws=skws)
        vmax = max(df[x].max(), df[y].max())
        vmin = min(df[x].min(), df[y].min())
        ax.set_xlim(180, vmax)
        ax.set_ylim(180, vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot((180, vmax), (180, vmax), color="dodgerblue", ls="--", lw=2)
        r, p = pearsonr(df[x].values, df[y].values)
        txt = "$r_{{xy}}$={0:.2f}\n$p$<0.05".format(r)
        ax.text(0.95, 0.05, txt, fontsize=10,
                va="bottom", zorder=100, transform=ax.transAxes, ha="right",
                bbox=box)
    #
    for k, ax in enumerate(axes):
        sns.despine(ax=ax)
        ax.set_aspect("equal")
        ax.text(0.05, 0.95, "(" + ascii_lowercase[k] + ")", fontsize=14,
                va="top", zorder=100, transform=ax.transAxes, ha="left",
                bbox=dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7))

    fig.tight_layout()
    plt.savefig(args.output, dpi=300, pad_inches=0.1,
                bbox_inches='tight')
    plt.show()



if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument("--input", "-i", nargs=3, action="store", dest="input",
                        required=True,
                        help="Input data MPL_PAR, MPL_SPC, CNN_SCP (.csv).",)

    parser.add_argument("--subset", "-s", action="store", dest="subset",
                        required=False, help="Either train, test or valid.",
                        default="test")

    # output figure
    parser.add_argument("--output", "-o", action="store", dest="output",
                        required=True, help="Output figure name (.png).",)

    args = parser.parse_args()

    main()
