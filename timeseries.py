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
    if len(args.input) != len(args.names):
        raise IOError("Input error. Please provide the same number of model "
                       "names as the number of input files.")

    # read and select
    dfs = []
    for inp, name in zip(args.input, args.names):
        df = pd.read_csv(inp, index_col="time")
        df["model"] = name
        dfs.append(df)
    df = pd.concat(dfs)

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
    colors = sns.color_palette("deep").as_hex()

    i = 1
    for model, mdf in df.groupby("model"):
        time = pd.to_datetime(mdf.index.values).to_pydatetime()
        ax1.scatter(time, mdf["Hs_buoy"], facecolor="k", edgecolor="k",
                    label="Observation", s=20)
        ax1.plot(time, mdf["Hs_wavewatch"], color=colors[0], label="WW3")
        ax1.plot(time, mdf["Hs_prediction"], label=model, color=colors[i])

        ax2.scatter(time, mdf["Tp_buoy"], facecolor="k", edgecolor="k",
                    label="Observation", s=20)
        ax2.plot(time, mdf["Tp_wavewatch"], color=colors[0])
        ax2.plot(time, mdf["Tp_prediction"], label=model, color=colors[i])

        ax3.scatter(time, mdf["Dm_buoy"], facecolor="k", edgecolor="k",
                    label="Observation", s=20)
        ax3.plot(time, mdf["Dm_wavewatch"], color=colors[0])
        ax3.plot(time, mdf["Dm_prediction"], label=model, color=colors[i])

        i += 1

    ax1.set_ylabel(r"$Hm_0$ $[m]$")
    ax2.set_ylabel(r"$Tm_{01}$ $[s]$")
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

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lg = fig.legend(by_label.values(), by_label.keys(),
                    ncol=len(args.names)+2, loc="upper center")
    lg.get_frame().set_color("w")

    fig.tight_layout()
    plt.savefig(args.output, dpi=300,
                pad_inches=0.1, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument("--input", "-i", nargs="*", action="store",
                        dest="input", required=True,
                        help="Input data (.csv).",)
    # input data
    parser.add_argument("--names", "-n", nargs="*", action="store",
                        dest="names", required=False, default=["ML"],
                        help="Model names.",)

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
