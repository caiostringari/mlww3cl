"""
Plot timeseries.

Use timeseries.py --help for usage.
"""
import argparse

import datetime

import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {"axes.linewidth": 2,
                        "legend.frameon": True,
                        "axes.facecolor": "w",
                        "grid.color": "w"})
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams['patch.edgecolor'] = "k"


def main():
    """Excute the main program."""
    # read and select
    df = pd.read_csv(args.input, index_col="time")
    locations = list(df["location"].unique())

    df = df.loc[df["location"] == args.loc_id]
    if df.empty:
        raise ValueError("Empty dataframe. Check location ID. Options are : " +
                         "\', \'".join(locations))

    # select a subset
    if args.subset != "all":
        df = df.loc[df["subset"] == args.subset]

    # sort
    df.sort_index(inplace=True)
    time = pd.to_datetime(df.index.values).to_pydatetime()

    # parse start date
    if args.start == "all":
        t1 = time[0]
    else:
        t1 = pd.to_datetime(args.start)

    # parse duration
    if args.dt == "all":
        t2 = time[-1]
    else:
        delta = datetime.timedelta(days=int(args.dt))
        t2 = t1 + delta

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax1.plot(time, df["Hs_buoy"], color="0.5", label="Observation")
    ax1.plot(time, df["Hs_wavewatch"], color="#ff4100", label="WW3")
    ax1.plot(time, df["Hs_prediction"], color="#018b8b", label="ML")
    ax1.set_ylabel(r"$Hm_0$ $[m]$")

    ax2.plot(time, df["Tp_buoy"], color="0.5")
    ax2.plot(time, df["Tp_wavewatch"], color="#ff4100")
    ax2.plot(time, df["Tp_prediction"], color="#018b8b")
    ax2.set_ylabel(r"$Tm_{01}$ $[s]$")

    ax3.plot(time, df["Dm_buoy"], color="0.5")
    ax3.plot(time, df["Dm_wavewatch"], color="#ff4100")
    ax3.plot(time, df["Dm_prediction"], color="#018b8b")
    ax3.set_ylabel(r"$Dm$ $[^o]$")

    # set axes
    locator = mdates.AutoDateLocator(minticks=10, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in [ax1, ax2, ax3]:

        sns.despine(ax=ax)
        ax.set_xlim(t1, t2)

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(12)
        ax.xaxis.get_offset_text().set_size(12)

    lg = fig.legend(ncol=3, loc="lower center")
    lg.get_frame().set_color("w")

    fig.tight_layout()
    plt.savefig(args.output, dpi=300,
                pad_inches=0.1, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument("--input", "-i", action="store", dest="input",
                        required=True, help="Input data (.csv).",)

    parser.add_argument("--subset", "-s", action="store", dest="subset",
                        required=False, help="Data subset (train, test, val).",
                        default="all")

    # location ID
    parser.add_argument("--location", "-ID", action="store", dest="loc_id",
                        required=True, help="Location ID.",)

    # start date
    parser.add_argument("--start", "-t1", action="store", dest="start",
                        required=False, default="all", help="Start date. "
                        "Use a format that pandas understands",)

    # duration
    parser.add_argument("--duration", "-dt", action="store", dest="dt",
                        required=False, help="Duration in days.", default="all")

    # output data
    parser.add_argument("--output", "-o", action="store", dest="output",
                        required=True, help="Output figure name (.png).",)

    args = parser.parse_args()

    main()
