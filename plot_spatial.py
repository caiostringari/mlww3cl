import pickle
import argparse

import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from cmocean import cm
from scipy.spatial import KDTree
from string import ascii_lowercase
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def linear_interp(meshx, meshy, x, y, z):
    """interpolate model results to ww3 mesh."""
    tree = KDTree(np.vstack([x, y]).T)
    dd, ii = tree.query(np.vstack([meshx, meshy]).T, k=[1])
    znew = []
    for i in ii:
        znew.append(float(z[i]))
    return znew


def add_colorbar(fig, ax, m, label):
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(
        size="3%", pad=0.3, axes_class=plt.Axes, pack_start=True
    )
    fig.add_axes(ax_cb)
    cb = plt.colorbar(m, cax=ax_cb, orientation="horizontal", extend="both")
    cb.set_label(label)


def get_mesh():
    """read the mesh"""
    ds = xr.open_dataset("mesh.nc")
    tri = ds["tri"].values
    trip = tri - tri / tri
    triang = mtri.Triangulation(ds.longitude.values, ds.latitude.values, trip)
    return triang


def main():

    # load data
    with open(args.input, 'rb') as handle:
        inp = pickle.load(handle)
    longitude = inp["longitude"]
    latitude = inp["latitude"]

    # load instruments location
    df = pd.read_csv("instrument_locations.csv")

    # means
    hs_diff = inp["hs_diff"].mean(axis=0)
    tp_diff = inp["tp_diff"].mean(axis=0)
    dm_diff = inp["dir_diff"].mean(axis=0)

    # load mesh
    triang = get_mesh()

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 3)

    # plot
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())

    # set mat extent
    max_lon = longitude.max()
    min_lon = longitude.min()
    max_lat = -32
    min_lat = -38
    for ax in [ax1, ax2, ax3]:
        ax.set_extent([max_lon, min_lon, max_lat, min_lat])

    # plot buoy locations
    for ax in [ax1, ax2, ax3]:
        for r, row in df.iterrows():
            if row["instrument"] == "Buoy":
                color = "orangered"
            else:
                color = "deepskyblue"
            ax.scatter(
                float(360 - row["longitude"]),
                float(-row["latitude"]),
                s=45,
                zorder=100,
                color=color,
                edgecolor="w",
            )

    # HS
    ihs = linear_interp(triang.x, triang.y, longitude, latitude, hs_diff)
    m = ax1.tripcolor(triang, ihs, vmin=-1.5, vmax=1.5, cmap=cm.curl)
    # ax1.set_title("Sig. Wave Height ($H_{m0}$) $[m]$")
    add_colorbar(fig, ax1, m, r"WWIII $H_{m0}$ - $MLP_{par}$ $H_{m0}$ [m]")

    # Tm01
    itm = linear_interp(triang.x, triang.y, longitude, latitude, tp_diff)
    m = ax2.tripcolor(triang, itm, cmap=cm.curl, vmin=-1, vmax=1)
    # ax2.set_title("Mean Wave Period ($T_{m01}$) $[s]$")
    add_colorbar(fig, ax2, m, r"WWIII $T_{m01}$ - $MLP_{par}$ $T_{m01}$ [s]")

    # Dp
    idm = linear_interp(triang.x, triang.y, longitude, latitude, dm_diff)
    m = ax3.tripcolor(triang, idm, cmap="viridis", vmin=0, vmax=45)
    ax3.set_title("Mean Wave Direcion ($D_m$) $[^{o}]$")
    add_colorbar(fig, ax3, m, r"WWIII $D_{m}$ - $MLP_{par}$ $D_{m}$ [$^o$]")

    # draw map elements
    lon_formatter = LongitudeFormatter(
        number_format=".1f", degree_symbol="", dateline_direction_label=True
    )
    lat_formatter = LatitudeFormatter(number_format=".1f", degree_symbol="")
    k = 0
    for ax in [ax1, ax2, ax3]:
        # ax.set_global()
        ax.coastlines(resolution="auto", color="k")
        ax.add_feature(cfeature.LAND, color="0.3", zorder=10)

        ax.set_xticks(np.round(np.arange(min_lon, max_lon, 2)), crs=ccrs.PlateCarree())
        ax.set_yticks(np.round(np.arange(min_lat, max_lat, 2)), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.text(
            0.05,
            0.975,
            "(" + ascii_lowercase[k] + ")",
            fontsize=14,
            va="top",
            zorder=100,
            transform=ax.transAxes,
            ha="left",
            bbox=dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7),
        )
        k += 1

    fname = "spatial_plot.png"
    plt.savefig(
        fname,
        dpi=300,
        transparent=False,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()
    plt.close()



if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument(
        "--input", "-i",
        action="store",
        dest="input",
        required=True,
        help="Results from predict_spatial.py.",
    )

    parser.add_argument(
        "--output", "-o",
        action="store",
        dest="output",
        required=True,
        help="Output figure name (.png)."
    )

    args = parser.parse_args()

    main()
