#
import numpy as np
import healpy as hp
import astropy.io.fits as pyfits
from multiprocessing import Pool
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from quicksipManera import *

dir = '$HOME/' # obviously needs to be changed
inputdir = '/project/projectdirs/cosmo/data/legacysurvey/dr3/' # where I get my data 
localdir = '/global/homes/m/manera/DESI/validation-outputs/' #place for local DESI stuff

extmap = np.loadtxt('/global/homes/m/manera/DESI/validation-outputs/healSFD_r_256_fullsky.dat') # extintion map remove it 

### A couple of useful conversions

def zeropointToScale(zp):
	return 10.**((zp - 22.5)/2.5)	

def nanomaggiesToMag(nm):
	return -2.5 * (log(nm,10.) - 9.)

def Magtonanomaggies(m):
	return 10.**(-m/2.5+9.)
	#-2.5 * (log(nm,10.) - 9.)


### Plotting facilities


def plotdepthfromIvar(band,depthmin=23.8,mjdmax='',prop='ivar',op='total',survey='DECaLS_DR3',nside='1024',oversamp='1'):
	import fitsio
	from matplotlib import pyplot as plt
	import matplotlib.cm as cm
	from numpy import zeros,array
	import healpix
	
	from healpix import pix2ang_ring,thphi2radec
	h = healpix.healpix()
	import healpy as hp

	f = fitsio.read(localdir+survey+mjdmax+'/nside'+nside+'_oversamp'+oversamp+'/'+survey+mjdmax+'_band_'+band+'_nside'+nside+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz')
	ral = []
	decl = []
	val = f['SIGNAL']
	magval = []
	for i in range(0,len(val)):
		magval.append(nanomaggiesToMag(sqrt(1./val[i]) * 5.))
	print min(magval),max(magval),len(f)/(float(nside)**2.*12)*360*360./pi
	if band == 'g':
		extc = 3.303/2.751
	if band == 'r':
		extc = 2.285/2.751
	if band == 'z':
		extc = 1.263/2.751
	magvalgmin = []
	for i in range(0,len(f)):
		#th,phi = pix2ang_ring(4096,f[i]['PIXEL'])
		th,phi = hp.pix2ang(int(nside),f[i]['PIXEL'])
		ra,dec = thphi2radec(th,phi)
		pix256 = h.ang2pix_nest(256,th,phi)
		magval[i] -= extmap[pix256]
		if magval[i] > depthmin:
			ral.append(ra)
			decl.append(dec)
			magvalgmin.append(magval[i])
	print len(ral)/(float(nside)**2.*12)*360*360./pi		
	#print min(val),max(val)	
	#map = plt.scatter(ral,decl,c=magvalgmin,cmap=cm.rainbow,lw=0,s=.1)
	map = plt.scatter(ral,decl,c=magvalgmin, cmap=cm.rainbow,s=2., vmin=21.0, vmax=23.5, lw=0,edgecolors='none')
	cbar = plt.colorbar(map)
	cbar.set_label(r'5$\sigma$ galaxy depth', rotation=270)
	plt.xlabel('r.a. (degrees)')
	plt.ylabel('declination (degrees)')
	plt.title('Map of depth' +' for '+survey+' '+band+'-band')
	plt.xlim(0,360)
	plt.ylim(-90,90)
	#plt.show()
	plt.savefig(localdir+'depth_'+band+'_'+survey+str(nside)+'.png')
	#plt.xscale('log')
	#pp.savefig()
	#pp.close()
	return True
		


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
#catalogue_name = 'DECaLS_DR2'+mjdw
pixoffset = 0 # How many pixels are being removed on the edge of each CCD? 15 for DES.
fname = inputdir+'ccds-annotated-decals.fits.gz'
#fname = localdir+'decals-ccds-annotated.fits'
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
    #('mjd_obs', '', 'max'), #
    ('ivar', '', 'total'),
    #('FWHM_FROMFLUXRADIUS_MEAN', '', 'mean'),
    #('FWHM_FROMFLUXRADIUS_MEAN', 'coaddweights3', 'mean'),
    #('NSTARS_ACCEPTED_MEAN', '', 'mean'),
    #('NSTARS_ACCEPTED_MEAN', 'coaddweights3', 'mean'),
    #('FWHM', '', 'mean'),
    #('FWHM', '', 'min'),
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
	+ ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height'] + ['ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3','ra','dec'] #, 'NAXIS1', 'NAXIS2']  \
	 #\
#    + ['PV1_'+str(i) for i in range(11)] + ['PV2_'+str(i) for i in range(11)] \
#   + ['RALL', 'RAUL', 'RAUR', 'RALR', 'DECLL', 'DECUL', 'DECUR', 'DECLR', 'URALL', 'UDECLL', 'URAUR', 'UDECUR', 'COADD_RA', 'COADD_DEC'] \
#    + ['COADD_MAGZP', 'MAGZP']

# Create big table with all relevant properties. We will send it to the Quicksip library, which will do its magic.
#tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [URAUL, UDECUL, URALR, UDECLR,ivar], names = propertiesToKeep + ['URAUL', 'UDECUL', 'URALR', 'UDECLR', 'ivar'])
tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar], names = propertiesToKeep + [ 'ivar'])

# Do the magic! Read the table, create Healtree, project it into healpix maps, and write these maps.
project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
#project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside)

# ------------------------------------------------------
#plot depth maps
# __________________

depthminl = [22.] #if doing multiple bands, add numbers there
for i in range(0,len(sample_names)):
	band = sample_names[i].split('_')[-1] 
	depthmin = depthminl[i]
	plotdepthfromIvar(band,depthmin=depthmin,mjdmax=mjdw,survey=catalogue_name)
