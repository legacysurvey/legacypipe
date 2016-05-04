
import numpy as np
import healpy as hp
import pyfits
from multiprocessing import Pool
from DESIccd import Magtonanomaggies

from quicksip import *

dir = '$HOME/' # obviously needs to be changed
localdir = '/Users/ashleyross/DESI/' #place for local DESI stuff


# ------------------------------------------------------
# DESI_RUN edited from Boris' DES example
# ------------------------------------------------------
# Main runs for SVA1 and Y1A1, from the IMAGE tables.
# ------------------------------------------------------

nside = 4096 # Resolution of output maps
nsidesout = None # if you want full sky degraded maps to be written
ratiores = 4 # Superresolution/oversampling ratio
mode = 1 # 1: fully sequential, 2: parallel then sequential, 3: fully parallel

catalogue_name = 'DECaLS_DR3'
pixoffset = 15 # How many pixels are being removed on the edge of each CCD? 15 for DES.
fname = localdir+'decals-ccds-annotated.fits'
# Where to write the maps ? Make sure directory exists.
outroot = localdir

tbdata = pyfits.open(fname)[1].data
print tbdata.dtype

# ------------------------------------------------------
# Read data

# Make sure we have the four corners of all coadd images.
# !!!AJR just put stuff in to make it run!!!
URALL = tbdata['ra0']
UDECLL = tbdata['dec0']
URAUR = tbdata['ra3']
UDECUR = tbdata['dec3']
COADD_RA = tbdata['ra']#(URALL+URAUR)/2
COADD_DEC = tbdata['dec']#(UDECLL+UDECUR)/2
m1 = np.divide(tbdata['ra3']-tbdata['ra0'],tbdata['dec3']-tbdata['dec0'])
m2 = (tbdata['ra2']-tbdata['ra1'])/(tbdata['dec2']-tbdata['dec1'])
ang = - np.arctan(np.abs((m1-m2)/(1+m1*m2)))
vecLL = [ (URALL - COADD_RA), (UDECLL - COADD_DEC) ]
vecUR = [ (URAUR - COADD_RA), (UDECUR - COADD_DEC) ]
URAUL = COADD_RA + np.cos(ang) * (URALL - COADD_RA) - np.sin(ang) * (UDECLL - COADD_DEC)
UDECUL = COADD_DEC + np.sin(ang) * (URALL - COADD_RA) + np.cos(ang) * (UDECLL - COADD_DEC)
URALR = COADD_RA + np.cos(ang) * (URAUR - COADD_RA) - np.sin(ang) * (UDECUR - COADD_DEC)
UDECLR = COADD_DEC + np.sin(ang) * (URAUR - COADD_RA) + np.cos(ang) * (UDECUR - COADD_DEC)

#def invnoisesq here

nmag = Magtonanomaggies(tbdata['galdepth'])/5.
ivar = 1./nmag**2.
# Select the five bands
indg = np.where((tbdata['filter'] == 'g') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) #& (tbdata['dec'] > 20.) & (tbdata['dec'] < 21.))
indr = np.where((tbdata['filter'] == 'r') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
indz = np.where((tbdata['filter'] == 'z') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
sample_names = ['band_g', 'band_r', 'band_z']
inds = [indg, indr, indz]

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
	+ ['RA', 'DEC', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'] + ['ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3'] #, 'NAXIS1', 'NAXIS2']  \
	 #\
#    + ['PV1_'+str(i) for i in range(11)] + ['PV2_'+str(i) for i in range(11)] \
#   + ['RALL', 'RAUL', 'RAUR', 'RALR', 'DECLL', 'DECUL', 'DECUR', 'DECLR', 'URALL', 'UDECLL', 'URAUR', 'UDECUR', 'COADD_RA', 'COADD_DEC'] \
#    + ['COADD_MAGZP', 'MAGZP']

# Create big table with all relevant properties. We will send it to the Quicksip library, which will do its magic.
tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [URAUL, UDECUL, URALR, UDECLR,ivar], names = propertiesToKeep + ['URAUL', 'UDECUL', 'URALR', 'UDECLR', 'ivar'])
print len(tbdata['ivar'])
# Fix ra/phi values beyond 360 degrees
#for nm in ['RALL', 'RAUL', 'RAUR', 'RALR', 'URALL', 'URAUR', 'COADD_RA']:
#    tbdata[nm] = np.mod(tbdata[nm], 360.0)

# Do the magic! Read the table, create Healtree, project it into healpix maps, and write these maps.
project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside, ratiores, pixoffset, nsidesout)

# ------------------------------------------------------
