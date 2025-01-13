#!/usr/bin/env python

"""Build and write out the NGC-star-clusters.fits catalog."""

import os
import numpy as np
import pandas as pd
import numpy.ma as ma
from astropy.io import ascii
from astropy.table import Table
from astropy.table import vstack as ap_vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from importlib.resources import files
import sys

sys.path.append("/Users/arielamsellem/Desktop/missing_PNe/legacypipe/")

# import desimodel.io
# import desimodel.footprint
# tiles = desimodel.io.load_tiles(onlydesi=True)

# Grab newest NGC catalog and its addendum
if not os.path.isfile("/tmp/NGC.csv"):
    os.system(
        "wget -P /tmp https://raw.githubusercontent.com/mattiaverga/OpenNGC/refs/heads/master/database_files/NGC.csv"
    )
if not os.path.isfile("/tmp/addendum.csv"):
    os.system(
        "wget -P /tmp https://raw.githubusercontent.com/mattiaverga/OpenNGC/refs/heads/master/database_files/addendum.csv"
    )

NGC = ascii.read("/tmp/NGC.csv", delimiter=";")
addendum = ascii.read("/tmp/addendum.csv", delimiter=";")

# Avoid merge conflict
conflict_cols = ["NGC", "IC", "Cstar Names"]
NGC.remove_columns(conflict_cols)
addendum.remove_columns(conflict_cols)
full_cat = ap_vstack([NGC, addendum])
# Mask out objects without coordinates
full_cat = full_cat[(full_cat["RA"] != "N/A")]

# Reformat ra, dec, and type
full_cat_coords = SkyCoord(full_cat["RA"], full_cat["Dec"], unit=(u.hourangle, u.deg))
full_cat["ra"] = full_cat_coords.ra.deg
full_cat["dec"] = full_cat_coords.dec.deg
objtype = np.char.strip(ma.getdata(full_cat["Type"]))

# Keep all globular clusters and planetary nebulae
keeptype = ("PN", "GCl")
keep = np.zeros(len(full_cat), dtype=bool)
for otype in keeptype:
    ww = [otype == tt for tt in objtype]
    keep = np.logical_or(keep, ww)
print(f"{np.sum(keep)} planetary nebulae and globular clusters found in OpenNGC.")
clusters = full_cat[keep]

# Fill missing major axes with a nominal 0.4 arcmin (roughly works
# for NGC7009, which is the only missing PN in the footprint).
ma.set_fill_value(clusters["MajAx"], 0.4)
clusters["MajAx"] = ma.filled(clusters["MajAx"].data)

# indesi = desimodel.footprint.is_point_in_desi(tiles, ma.getdata(clusters['ra']), ma.getdata(clusters['dec']))
# print(np.sum(indesi))
# bb = clusters[indesi]
# bb[np.argsort(bb['majax'])[::-1]]['name', 'ra', 'dec', 'majax', 'type']

# Build the output catalog: select a subset of the columns and rename
# majax-->radius (arcmin-->degree)
out = Table()
out["name"] = clusters["Name"]
out["alt_name"] = [
    "" if mm == 0 else "M{}".format(str(mm)) for mm in ma.getdata(clusters["M"])
]
out["type"] = clusters["Type"]
out["ra"] = clusters["ra"]
out["dec"] = clusters["dec"]
out["radius"] = (clusters["MajAx"] / 60).astype("f4")  # [degrees]
# out['radius'] = out['radius_orig']
# add a position angle and ellipticity (b/a)
out["pa"] = np.zeros(len(out), dtype="f4")
out["ba"] = np.ones(len(out), dtype="f4")

# Read the updated radii based on visual inspection by Arjun Dey (Feb 2020):
radiifile = files("legacypipe").joinpath("data/NGC-star-clusters-radii.csv")
newname, newradii, newpa, newba = np.loadtxt(
    radiifile, dtype=str, delimiter=",", unpack=True
)
out["radius"][np.isin(out["name"], newname)] = newradii.astype("f4")
out["pa"][np.isin(out["name"], newname)] = newpa.astype("f4")
out["ba"][np.isin(out["name"], newname)] = newba.astype("f4")

# oldradii = out['radius'].copy()
# import matplotlib.pyplot as plt
# plt.scatter(oldradii*60, oldradii/out['radius'], s=15)
# plt.xlabel('Old radii [arcmin]')
# plt.ylabel('Old radii / New radii')
# plt.show()

# Read the ancillary globular cluster catalog and update the radii in the NGC.
# https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3table.pl?tablehead=name%3Dglobclust&Action=More+Options
if False:
    from astrometry.libkd.spherematch import match_radec

    gcfile = files("legacypipe").joinpath("data/globular_clusters.fits")
    gcs = Table.read(gcfile)
    I, J, _ = match_radec(
        clusters["ra"],
        clusters["dec"],
        gcs["RA"],
        gcs["DEC"],
        10.0 / 3600.0,
        nearest=True,
    )
    out["radius"][I] = (gcs["HALF_LIGHT_RADIUS"][J] / 60).astype("f4")  # [degrees]

# Read the supplemental catalog of globular clusters and (compact) open clusters
# from Arjun Dey (Mar 2020). Note that the NGC open clusters were culled above,
# but we put them back here because the diameters have been vetted.
suppfile = files("legacypipe").joinpath("data/star-clusters-supplemental.csv")
supp = Table.read(suppfile, delimiter=",")
supp["alt_name"] = supp["alt_name"].astype("U4")
out = ap_vstack((out, supp))

if False:  # debugging
    bb = out[["M" in nn for nn in out["alt_name"]]]
    bb[np.argsort(bb["radius"])]
    bb["radius"] *= 60
    bb["radius_orig"] *= 60
    print(bb)

# Read in the HASH catalog
hash_file = files("legacypipe").joinpath("data/HASH_PNe_geometry.csv")
hash_tab = pd.read_csv(hash_file, index_col=0)

# Remove PNe that are in HASH from the out table
req_min_sep = 5.0  # arcsec
out_coords = SkyCoord(out["ra"] << u.deg, out["dec"] << u.deg)
hash_coords = SkyCoord(hash_tab["ra"] << u.deg, hash_tab["dec"] << u.deg)
match_idxs, match_seps, _ = match_coordinates_sky(out_coords, hash_coords)
duplicate_mask = ~(match_seps.arcsec < req_min_sep)
out = out[duplicate_mask]

# Write PNe and GCls to file
out = out[np.argsort(out["ra"])]
clusterfile = files("legacypipe").joinpath("data/NGC-star-clusters.fits")
print("Writing {}".format(clusterfile))
out.write(clusterfile, overwrite=True)

# Code to help visually check all the globular clusters.
if False:
    checktype = ("GCl", "PN")
    check = np.zeros(len(full_cat), dtype=bool)
    for otype in checktype:
        ww = [otype == tt for tt in objtype]
        check = np.logical_or(check, ww)
    check_clusters = full_cat[check]  # 845 of them

    # Write out a catalog, load it into the viewer and look at each of them.
    check_clusters[["ra", "dec", "Name"]].write(
        "/tmp/check.fits", overwrite=True
    )  # 25 of them
