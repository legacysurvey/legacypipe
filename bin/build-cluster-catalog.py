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

# import desimodel.io
# import desimodel.footprint
# tiles = desimodel.io.load_tiles(onlydesi=True)

# Grab newest NGC catalog and its addendum
if not os.path.isfile("/tmp/NGC.csv"):
    os.system(
        "wget -P /tmp https://raw.githubusercontent.com/mattiaverga/OpenNGC/refs/tags/v20231203/database_files/NGC.csv"
    )
if not os.path.isfile("/tmp/addendum.csv"):
    os.system(
        "wget -P /tmp https://raw.githubusercontent.com/mattiaverga/OpenNGC/refs/tags/v20231203/database_files/addendum.csv"
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

# Read the updated radii based on visual inspection by Arjun Dey (Feb 2020) with updates by Ariel Amsellem (Jul 2025):
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
# Add HASH catalog objects that are outside the DR10 footprint
hash_noimage_file = files("legacypipe").joinpath("data/HASH_PNe_outside_footprint.csv")
noimage_tab = pd.read_csv(hash_noimage_file, index_col=0)
hash_tab = pd.concat([hash_tab, noimage_tab], axis=0)

# Separate the OpenNGC objects into PNe and GCls
out_pne = out[out["type"] == "PN"]
out_gcl = out[out["type"] == "GCl"]

# Remove PNe that are in HASH from the out table
req_min_sep = 5.5  # arcsec
out_pne_coords = SkyCoord(out_pne["ra"] << u.deg, out_pne["dec"] << u.deg)
hash_coords = SkyCoord(hash_tab["ra"] << u.deg, hash_tab["dec"] << u.deg)
match_idxs, match_seps, _ = match_coordinates_sky(out_pne_coords, hash_coords)
# Note: Many PNe in OpenNGC were left out of the HASH table because they were not
# in the legacy survey footprint and therefore could not be visualy verified.
# These PNe are still included in the out table
duplicate_mask = ~(match_seps.arcsec < req_min_sep)
out_pne = out_pne[duplicate_mask]

# Restructure the HASH catalog so that it can be appended to the out tables
hash_tab = Table.from_pandas(hash_tab)
# Mask PNe that were assigned major/minor axis ratios of 0.0 arcsec
# (This was done either because they couldn't be seen in the Legacy Survey)
hash_tab = hash_tab[(hash_tab["major_axis"] > 0.0) & (hash_tab["minor_axis"] > 0.0)]
# Convert from arcseconds to degrees
hash_tab["major_axis"] /= 3600.0
hash_tab["minor_axis"] /= 3600.0
hash_tab["ba"] = hash_tab["minor_axis"] / hash_tab["major_axis"]
hash_tab["alt_name"] = "--"
hash_tab["type"] = "PN"
hash_tab["radius"] = hash_tab["major_axis"] / 2
# Uniformly inflate all HASH PNe radii by 15%
# hash_tab["radius"] *= 1.15
hash_tab_keep = hash_tab[
    ["name", "alt_name", "type", "ra", "dec", "radius", "pa", "ba"]
]

# Read in catalog of known and visually verified Reflection Nebulae (RNe)
# (See https://heasarc.gsfc.nasa.gov/w3browse/all/refnebulae.html for original catalog)
rne_file = files("legacypipe").joinpath("data/RNe_geometry.csv")
rne_tab = pd.read_csv(rne_file, index_col=0)
# Mask RNe that were assigned major axis ratios of 0.0 arcsec
# (This was done either because they couldn't be seen in the Legacy Survey or were part of larger RNe)
rne_tab = rne_tab[(rne_tab["major_axis"] > 0.0) & (rne_tab["minor_axis"] > 0.0)]
# Restructure the RNe catalog so that it can be appended to the out tables
rne_tab["major_axis"] /= 3600.0
rne_tab["minor_axis"] /= 3600.0
rne_tab["ba"] = rne_tab["minor_axis"] / rne_tab["major_axis"]
# Names are from the original catalog column 'Seq'
rne_tab["name"] = rne_tab["name"].astype(str)
rne_tab["alt_name"] = "--"
rne_tab["type"] = "RN"
rne_tab["radius"] = rne_tab["major_axis"] / 2
rne_tab_keep = rne_tab[["name", "alt_name", "type", "ra", "dec", "radius", "pa", "ba"]]
# Convert the RNe pandas DataFrame to an Astropy Table (to be compatible with the other output tables)
rne_tab_keep = Table.from_pandas(rne_tab_keep)

# Combine all clusters, PNe, and RNe into a single table
print(f"Using {len(out_gcl)} globular clusters found in OpenNGC.")
print(f"Using {len(out_pne)} planetary nebulae found in OpenNGC.")
print(f"Using {len(hash_tab_keep)} planetary nebulae found in HASH.")
print(f"Using {len(rne_tab_keep)} reflection nebulae found at HEASARC.")
out = ap_vstack([out_gcl, out_pne, hash_tab_keep, rne_tab_keep])

# Make a minimum size cut on all objects of 10 arcseconds in radius
min_size_deg = 10.0 / 3600.0
out = out[out["radius"] >= min_size_deg]
# Offset all PA values by 90 degrees (to be compatible with the Legacy Survey convention)
out["pa"][out["pa"] != 0.0] = (180.0 - out["pa"][out["pa"] != 0.0]) % 180.0

# Convert the output table to a pandas DataFrame for easier manipulation
out = out.to_pandas()

# For globular clusters, insert a space between NGC/IC and identifier numbers
# Also, remove leading zeros from identifiers and trailing en-dashes
ngc_mask = out["name"].notna() & out["name"].str.startswith("NGC")
out.loc[ngc_mask, "name"] = (
    "NGC "
    + out.loc[ngc_mask, "name"]
    .str[3:]
    .str.split("-")
    .str[0]
    .str.replace(r"^0+", " ", regex=True)
).str.replace(r"NGC\s+", "NGC ", regex=True)
ic_mask = out["name"].notna() & out["name"].str.startswith("IC")
out.loc[ic_mask, "name"] = (
    "IC "
    + out.loc[ic_mask, "name"]
    .str[2:]
    .str.split("-")
    .str[0]
    .str.replace(r"^0+", " ", regex=True)
).str.replace(r"IC\s+", "IC ", regex=True)

# For deisgnations beginning with "ESO", ensure there is a space between ESO and the numeric identifier
eso_mask = out["name"].notna() & out["name"].astype(str).str.startswith("ESO")
eso_no_space_mask = eso_mask & (~out["name"].astype(str).str.startswith("ESO "))
out.loc[eso_no_space_mask, "name"] = (
    "ESO " + out.loc[eso_no_space_mask, "name"].astype(str).str[3:]
).str.encode("utf-8")

# For deisgnations beginning with "MWSC", add '[KPS2012] ' to the beginning of the designation (making the name SIMBAD compatible)
mwsc_mask = (out["type"] == "GCl") & (out["name"].str.startswith("MWSC"))
out.loc[mwsc_mask, "name"] = "[KPS2012] " + out.loc[mwsc_mask, "name"].astype(str)

# Convert back to an astropy table
out = Table.from_pandas(out)

# Write PNe and GCls to file
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
