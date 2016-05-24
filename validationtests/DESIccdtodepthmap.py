#
import numpy as np
import healpy as hp
import pyfits
from multiprocessing import Pool
from DESIccd import Magtonanomaggies
import matplotlib.pyplot as plt

from quicksip import *

dir = '$HOME/' # obviously needs to be changed
localdir = '/Users/ashleyross/DESI/' #place for local DESI stuff


# ------------------------------------------------------
# AJR hack of Boris' code. Many original things from Boris are just commented out
# DESI_RUN edited from Boris' DES example
# ------------------------------------------------------

nside = 1024 # Resolution of output maps
nsidesout = None # if you want full sky degraded maps to be written
ratiores = 1 # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
mode = 1 # 1: fully sequential, 2: parallel then sequential, 3: fully parallel

mjd_max = 10e10
#mjd_max = 57000
mjdw = ''
if mjd_max != 10e10:
	mjdw += 'mjdmax'+str(mjd_max)
catalogue_name = 'DECaLS_DR3'+mjdw
pixoffset = 0 # How many pixels are being removed on the edge of each CCD? 15 for DES.
fname = localdir+'ccds-annotated-decals.fits.gz'
# Where to write the maps ? Make sure directory exists.
outroot = localdir

tbdata = pyfits.open(fname)[1].data
print tbdata.dtype

# ------------------------------------------------------
# Read data

#def invnoisesq here
nmag = Magtonanomaggies(tbdata['galdepth'])/5.
ivar = 1./nmag**2.
# Select the five bands
#indg = np.where((tbdata['filter'] == 'g') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True) & (tbdata['mjd_obs'] < mjd_max))# & (tbdata['ra'] < 50.))# & (tbdata['dec'] < 21.))
#indr = np.where((tbdata['filter'] == 'r') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
indz = np.where((tbdata['filter'] == 'z') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
#sample_names = ['band_g', 'band_r', 'band_z']
sample_names = ['band_z']
#inds = [indg, indr, indz]
inds = [indz]

# What properties do you want mapped?
# Each each tuple, the first element is the quantity to be projected,
# the second is the weighting scheme, and the third is the operation.
propertiesandoperations = [
    #('count', '', 'fracdet'),  # CCD count with fractional values in pixels partially observed.
    #('EXPTIME', '', 'total'), # Total exposure time, constant weighting.
    #('maglimit3', '', ''), # Magnitude limit (3rd method, correct)
    #('FWHM_MEAN', '', 'mean'), # Mean FWHM (called FWHM_MEAN because it's already the mean per CCD)
    #('FWHM_MEAN', 'coaddweights3', 'mean'), # Same with with weighting corresponding to CDD noise
    ('mjd_obs', '', 'max'), #
    ('ivar', '', 'total'),
    #('FWHM_FROMFLUXRADIUS_MEAN', '', 'mean'),
    #('FWHM_FROMFLUXRADIUS_MEAN', 'coaddweights3', 'mean'),
    #('NSTARS_ACCEPTED_MEAN', '', 'mean'),
    #('NSTARS_ACCEPTED_MEAN', 'coaddweights3', 'mean'),
    ('FWHM', '', 'mean'),
    ('FWHM', '', 'min'),
    #('AIRMASS', '', 'mean'),
    #('AIRMASS', 'coaddweights3', 'mean'),
    #('SKYBRITE', '', 'mean'),
    #('SKYBRITE', 'coaddweights3', 'mean'),
    #('SKYSIGMA', '', 'mean'),
    #('SKYSIGMA', 'coaddweights3', 'mean')
    ]

# What properties to keep when reading the images? Should at least contain propertiesandoperations and the image corners.
#propertiesToKeep = ['COADD_ID', 'ID', 'TILENAME', 'BAND', 'AIRMASS', 'SKYBRITE', 'SKYSIGMA', 'EXPTIME'] \
#	+ ['FWHM', 'FWHM_MEAN', 'FWHM_PIXELFREE_MEAN', 'FWHM_FROMFLUXRADIUS_MEAN', 'NSTARS_ACCEPTED_MEAN'] \
propertiesToKeep = [ 'filter', 'AIRMASS', 'FWHM','mjd_obs'] \
	+ ['RA', 'DEC', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'] + ['ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3','ra','dec'] #, 'NAXIS1', 'NAXIS2']  \
	 #\
#    + ['PV1_'+str(i) for i in range(11)] + ['PV2_'+str(i) for i in range(11)] \
#   + ['RALL', 'RAUL', 'RAUR', 'RALR', 'DECLL', 'DECUL', 'DECUR', 'DECLR', 'URALL', 'UDECLL', 'URAUR', 'UDECUR', 'COADD_RA', 'COADD_DEC'] \
#    + ['COADD_MAGZP', 'MAGZP']

# Create big table with all relevant properties. We will send it to the Quicksip library, which will do its magic.
#tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [URAUL, UDECUL, URALR, UDECLR,ivar], names = propertiesToKeep + ['URAUL', 'UDECUL', 'URALR', 'UDECLR', 'ivar'])
tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar], names = propertiesToKeep + [ 'ivar'])

# Do the magic! Read the table, create Healtree, project it into healpix maps, and write these maps.
#project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside)

# ------------------------------------------------------
#plot depth maps
# __________________

from DESIccd import plotdepthfromIvar
depthminl = [23.] #if doing multiple bands, add numbers there
for i in range(0,len(sample_names)):
	band = sample_names[i].split('_')[-1] 
	depthmin = depthminl[i]
	plotdepthfromIvar(band,depthmin=depthmin,mjdmax=mjdw)
