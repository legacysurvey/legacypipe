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
#inputdir = '/project/projectdirs/cosmo/data/legacysurvey/dr3/' # where I get my data 
inputdir= '/global/projecta/projectdirs/cosmo/work/dr4/'
localdir = '/global/homes/m/manera/DESI/validation-outputs/' #place for local DESI stuff

#extmap = np.loadtxt('/global/homes/m/manera/DESI/validation-outputs/healSFD_r_256_fullsky.dat') # extintion map remove it 



### A couple of useful conversions

def zeropointToScale(zp):
	return 10.**((zp - 22.5)/2.5)	

def nanomaggiesToMag(nm):
	return -2.5 * (log(nm,10.) - 9.)

def Magtonanomaggies(m):
	return 10.**(-m/2.5+9.)
	#-2.5 * (log(nm,10.) - 9.)


### Plotting facilities

		
def plotPropertyMap(band,vmin=21.0,vmax=24.0,mjdmax='',prop='ivar',op='total',survey='surveyname',nside='1024',oversamp='1'):
	import fitsio
	from matplotlib import pyplot as plt
	import matplotlib.cm as cm
	from numpy import zeros,array
	import healpix
	
	from healpix import pix2ang_ring,thphi2radec
	import healpy as hp

        fname=localdir+survey+mjdmax+'/nside'+nside+'_oversamp'+oversamp+'/'+survey+mjdmax+'_band_'+band+'_nside'+nside+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
        f = fitsio.read(fname)

	ral = []
	decl = []
	val = f['SIGNAL']
        pix = f['PIXEL']
	
        # Obtain values to plot 
        if (prop == 'ivar'):
	    myval = []
            mylabel='depth' 
            print 'Converting ivar to depth.'
            print 'Plotting depth'
            
            below=0 
            for i in range(0,len(val)):
                depth=nanomaggiesToMag(sqrt(1./val[i]) * 5.)
                if(depth < vmin):
                     below=below+1
                else:
                    myval.append(depth)
                    th,phi = hp.pix2ang(int(nside),pix[i])
	            ra,dec = thphi2radec(th,phi)
	            ral.append(ra)
	            decl.append(dec)

        npix=len(f)
        print 'Min and Max values of ', mylabel, ' values is ', min(myval), max(myval)
        print 'Number of pixels is ', npix
        print 'Number of pixels offplot with ', mylabel,' < ', vmin, ' is', below 
        print 'Area is ', npix/(float(nside)**2.*12)*360*360./pi, ' sq. deg.'


	map = plt.scatter(ral,decl,c=myval, cmap=cm.rainbow,s=2., vmin=vmin, vmax=vmax, lw=0,edgecolors='none')
	cbar = plt.colorbar(map)
	plt.xlabel('r.a. (degrees)')
	plt.ylabel('declination (degrees)')
	plt.title('Map of '+ mylabel +' for '+survey+' '+band+'-band')
	plt.xlim(0,360)
	plt.ylim(-30,90)
	plt.savefig(localdir+mylabel+'_'+band+'_'+survey+str(nside)+'.png')
        plt.close()
	#plt.show()
	#cbar.set_label(r'5$\sigma$ galaxy depth', rotation=270,labelpad=1)
    	#plt.xscale('log')
	return True
		

def depthfromIvar(band,rel='DR3',survey='survename'):
    
    # ------------------------------------------------------
    # MARCM stable version, improved from AJR quick hack 
    # This now included extinction from the exposures
    # Uses quicksip subroutines from Boris 
    # (with a bug I corrected for BASS and MzLS ccd orientation) 
    # Produces depth maps from Dustin's annotated files 
    # ------------------------------------------------------
    
    nside = 1024 # Resolution of output maps
    nsidesout = None # if you want full sky degraded maps to be written
    ratiores = 1 # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
    mode = 1 # 1: fully sequential, 2: parallel then sequential, 3: fully parallel
    
    pixoffset = 0 # How many pixels are being removed on the edge of each CCD? 15 for DES.
    
    mjd_max = 10e10
    mjdw = ''

    # Survey inputs
    if rel == 'DR2':
        fname =inputdir+'decals-ccds-annotated.fits'
        catalogue_name = 'DECaLS_DR2'+mjdw

    if rel == 'DR3':
        inputdir = '/project/projectdirs/cosmo/data/legacysurvey/dr3/' # where I get my data 
        fname =inputdir+'ccds-annotated-decals.fits.gz'
        catalogue_name = 'DECaLS_DR3'+mjdw

    if rel == 'DR4':
        inputdir= '/global/projecta/projectdirs/cosmo/work/dr4/'
        if (band == 'g' or band == 'r'):
            fname=inputdir+'ccds-annotated-dr4-90prime.fits.gz'
            catalogue_name = '90prime_DR4'+mjdw
        if band == 'z' :
            fname = inputdir+'ccds-annotated-dr4-mzls.fits.gz' 
            catalogue_name = 'MZLS_DR4'+mjdw
	
    #Bands inputs
    if band == 'g':
        be = 1
        extc = 3.303  #/2.751
    if band == 'r':
        be = 2
        extc = 2.285  #/2.751
    if band == 'z':
        be = 4
        extc = 1.263  #/2.751

    # Where to write the maps ? Make sure directory exists.
    outroot = localdir
    
    tbdata = pyfits.open(fname)[1].data
    
    # ------------------------------------------------------
    # Obtain indices
    if band == 'g':
        sample_names = ['band_g']
        indg = np.where((tbdata['filter'] == 'g') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
        inds = indg #redundant
    if band == 'r':
        sample_names = ['band_r']
        indr = np.where((tbdata['filter'] == 'r') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
        inds = indr #redundant
    if band == 'z':
        sample_names = ['band_z']
        indz = np.where((tbdata['filter'] == 'z') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
        inds = indz # redundant
    
    #Read data 
    #obtain invnoisesq here, including extinction 
    nmag = Magtonanomaggies(tbdata['galdepth']-extc*tbdata['EBV'])/5.
    ivar= 1./nmag**2.

    # What properties do you want mapped?
    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    propertiesandoperations = [ ('ivar', '', 'total'), ]

 
    # What properties to keep when reading the images? 
    #Should at least contain propertiesandoperations and the image corners.
    # MARCM - actually no need for ra dec image corners.   
    # Only needs ra0 ra1 ra2 ra3 dec0 dec1 dec2 dec3 only if fast track appropriate quicksip subroutines were implemented 
    propertiesToKeep = [ 'filter', 'AIRMASS', 'FWHM','mjd_obs'] \
    	+ ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height']
    
    # Create big table with all relevant properties. 

    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar], names = propertiesToKeep + [ 'ivar'])
    
    # Read the table, create Healtree, project it into healpix maps, and write these maps.
    # Done with Quicksip library, note it has quite a few hardcoded values (use new version by MARCM for BASS and MzLS) 
    # project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside)
    project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
 
    # ----- plot depth map -----
    prop='ivar'
    plotPropertyMap(band,survey=catalogue_name,prop=prop)

    return True


def plotMaghist_pred(band,FracExp=[0,0,0,0,0],ndraw = 1e5,nbin=100,rel='DR3',vmin=21.0):

    # MARCM Makes histogram of predicted magnitudes 
    # by MonteCarlo from exposures converving fraction of number of exposures
    # This produces the histogram for Dustin's processed galaxy depth

    import fitsio
    from matplotlib import pyplot as plt
    from numpy import zeros,array
    from random import random

    # Check fraction of number of exposures adds to 1. 
    if( abs(sum(FracExp) - 1.0) > 1e-5 ):
       print sum(FracExp)
       raise ValueError("Fration of number of exposures don't add to one")

    # Survey inputs
    mjdw='' 
    if rel == 'DR2':
        fname =inputdir+'decals-ccds-annotated.fits'
        catalogue_name = 'DECaLS_DR2'+mjdw

    if rel == 'DR3':
        inputdir = '/project/projectdirs/cosmo/data/legacysurvey/dr3/' # where I get my data 
        fname =inputdir+'ccds-annotated-decals.fits.gz'
        catalogue_name = 'DECaLS_DR3'+mjdw

    if rel == 'DR4':
        inputdir= '/global/projecta/projectdirs/cosmo/work/dr4/'
        if (band == 'g' or band == 'r'):
            fname=inputdir+'ccds-annotated-dr4-90prime.fits.gz'
            catalogue_name = '90prime_DR4'+mjdw
        if band == 'z' :
            fname = inputdir+'ccds-annotated-dr4-mzls.fits.gz' 
            catalogue_name = 'MZLS_DR4'+mjdw
		
    # Bands info 
    if band == 'g':
        be = 1
        zp0 = 25.08
        recm = 24.
    if band == 'r':
        be = 2
        zp0 = 25.29
        recm = 23.4
    if band == 'z':
        be = 4
        zp0 = 24.92
        recm = 22.5

    f = fitsio.read(fname)

    #read in magnitudes including extinction
    counts2014 =0
    n = 0
    nl = []
    for i in range(0,len(f)):
        DS = 0
        year = int(f[i]['date_obs'].split('-')[0])
        if (year <= 2014): counts2014=counts2014+1 
        if year > 2014:
            DS = 1 #enforce 2015 data

            if f[i]['filter'] == band:
			
                if DS == 1:
                    n += 1				
                    if f[i]['dec'] > -20 and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True :   
		
                         magext = f[i]['galdepth'] - f[i]['decam_extinction'][be]
                         nmag = Magtonanomaggies(magext)/5. #total noise
                         nl.append(nmag)

    ng = len(nl)
    print "-----------"
    print "Number of objects with DS=1", n
    print "Number of objects in the sample", ng 
    print "Counts before or during 2014", counts2014

    #Monte Carlo to predict magnitudes histogram 
    ndrawn = 0
    nbr = 0	
    NTl = []
    for indx, f in enumerate(FracExp,1) : 
        Nexp = indx # indx starts at 1 bc argument on enumearate :-), thus is the number of exposures
        nd = int(round(ndraw * f))
        ndrawn=ndrawn+nd

        for i in range(0,nd):

            detsigtoti = 0
            for j in range(0,Nexp):
		ind = int(random()*ng)
		detsig1 = nl[ind]
		detsigtoti += 1./detsig1**2.

 	    detsigtot = sqrt(1./detsigtoti)
	    m = nanomaggiesToMag(detsigtot * 5.)
	    if m > recm: # pass requirement
	 	nbr += 1.	
	    NTl.append(m)
	    n += 1.
	
    # Run some statistics 

    NTl=np.array(NTl)
    mean = sum(NTl)/float(len(NTl))
    std = sqrt(sum(NTl**2.)/float(len(NTl))-mean**2.)
    NTl.sort()
    if len(NTl)/2. != len(NTl)/2:
        med = NTl[len(NTl)/2+1]
    else:
        med = (NTl[len(NTl)/2+1]+NTl[len(NTl)/2])/2.

    print "Mean ", mean
    print "Median ", med
    print "Std ", std
    print 'percentage better than requirements '+str(nbr/float(ndrawn))

    # Prepare historgram 
    minN = max(min(NTl),vmin)
    maxN = max(NTl)+.0001
    hl = zeros((nbin)) # histogram counts
    lowcounts=0
    for i in range(0,len(NTl)):
        bin = int(nbin*(NTl[i]-minN)/(maxN-minN))
        if(bin >= 0) : 
            hl[bin] += 1
        else:
            lowcounts +=1

    Nl = []  # x bin centers
    for i in range(0,len(hl)):
        Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))
    NTl = array(NTl)

    #### Ploting histogram 
    print "Plotting the histogram now"
    print "min,max depth ",min(NTl), max(NTl) 
    print "counts below ", vmin, "are ", lowcounts

    from matplotlib.backends.backend_pdf import PdfPages
    plt.clf()
    pp = PdfPages(localdir+'validationplots/'+catalogue_name+band+'_pred_exposures.pdf')	

    plt.plot(Nl,hl,'k-')
    plt.xlabel(r'5$\sigma$ '+band+ ' depth')
    plt.ylabel('# of images')
    plt.title('MC combined exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(ndrawn))[:5]+'\n '+catalogue_name)
    #plt.xscale('log')
    pp.savefig()
    pp.close()
    return True

def photometricReq(band,rel='DR3',survey='survename'):
    
    # ------------------------------------------------------
    # ------------------------------------------------------
    
    nside = 1024 # Resolution of output maps
    nsidesout = None # if you want full sky degraded maps to be written
    ratiores = 1 # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
    mode = 1 # 1: fully sequential, 2: parallel then sequential, 3: fully parallel
    
    pixoffset = 0 # How many pixels are being removed on the edge of each CCD? 15 for DES.
    
    mjd_max = 10e10
    mjdw = ''

    # Survey inputs
    if rel == 'DR2':
        fname =inputdir+'decals-ccds-annotated.fits'
        catalogue_name = 'DECaLS_DR2'+mjdw

    if rel == 'DR3':
        inputdir = '/project/projectdirs/cosmo/data/legacysurvey/dr3/' # where I get my data 
        fname =inputdir+'ccds-annotated-decals.fits.gz'
        catalogue_name = 'DECaLS_DR3'+mjdw

    if rel == 'DR4':
        inputdir= '/global/projecta/projectdirs/cosmo/work/dr4/'
        if (band == 'g' or band == 'r'):
            fname=inputdir+'ccds-annotated-dr4-90prime.fits.gz'
            catalogue_name = '90prime_DR4'+mjdw
        if band == 'z' :
            fname = inputdir+'ccds-annotated-dr4-mzls.fits.gz' 
            catalogue_name = 'MZLS_DR4'+mjdw
	
    # Where to write the maps ? Make sure directory exists.
    outroot = localdir
    
    tbdata = pyfits.open(fname)[1].data
    
    # ------------------------------------------------------
    # Obtain indices
    if band == 'g':
        sample_names = ['band_g']
        indg = np.where((tbdata['filter'] == 'g') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
        inds = indg #redundant
    if band == 'r':
        sample_names = ['band_r']
        indr = np.where((tbdata['filter'] == 'r') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
        inds = indr #redundant
    if band == 'z':
        sample_names = ['band_z']
        indz = np.where((tbdata['filter'] == 'z') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
        inds = indz # redundant
    
    #Read data 
    #obtain invnoisesq here, including extinction 
    zptvar = tbdata['CCDPHRMS']**2/tbdata['CCDNMATCH']

    # What properties do you want mapped?
    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    propertiesandoperations = [ ('zptvar', '', 'total'), ('nccd','','total')]

 
    # What properties to keep when reading the images? 
    #Should at least contain propertiesandoperations and the image corners.
    # MARCM - actually no need for ra dec image corners.   
    # Only needs ra0 ra1 ra2 ra3 dec0 dec1 dec2 dec3 only if fast track appropriate quicksip subroutines were implemented 
    propertiesToKeep = [ 'filter', 'AIRMASS', 'FWHM','mjd_obs'] \
    	+ ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height']
    
    # Create big table with all relevant properties. 

    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [zptvar,nccd], names = propertiesToKeep + [ 'zptvar','nccd'])
    
    # Read the table, create Healtree, project it into healpix maps, and write these maps.
    # Done with Quicksip library, note it has quite a few hardcoded values (use new version by MARCM for BASS and MzLS) 
    # project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside)
    project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
 
    # ----- plot depth map -----
    #prop='ivar'
    #plotPropertyMap(band,survey=catalogue_name,prop=prop)

    return True


# ***********************************************************************
# ***********************************************************************

# --- run depth maps 
#band='r'
#depthfromIvar(band,rel='DR3')
#
#band='g'
#depthfromIvar(band,rel='DR3')
#
#band='z'
#depthfromIvar(band,rel='DR3')

#band='r'
#depthfromIvar(band,rel='DR4')

#band='g'
#depthfromIvar(band,rel='DR4')

#band='z'
#depthfromIvar(band,rel='DR4')



# DECALS (DR3) the final survey will be covered by 
# 1, 2, 3, 4, and 5 exposures in the following fractions: 
#FracExp=[0.02,0.24,0.50,0.22,0.02]

#print "DECaLS depth histogram r-band"
#band='r'
#plotMaghist_pred(band,FracExp=FracExp,ndraw = 1e5,nbin=100,rel='DR3')

#print "DECaLS depth histogram g-band"
#band='g'
#plotMaghist_pred(band,FracExp=FracExp,ndraw = 1e5,nbin=100,rel='DR3')

#print "DECaLS depth histogram z-band"
#band='z'
#plotMaghist_pred(band,FracExp=FracExp,ndraw = 1e5,nbin=100,rel='DR3')

# For BASS (DR4) the coverage fractions for 1,2,3,4,5 exposures are:
#FracExp=[0.0014,0.0586,0.8124,0.1203,0.0054,0.0019]

#print "BASS depth histogram r-band"
#band='r'
#plotMaghist_pred(band,FracExp=FracExp,ndraw = 1e5,nbin=100,rel='DR4')

#print "BASS depth histogram g-band"
#band='g'
#plotMaghist_pred(band,FracExp=FracExp,ndraw = 1e5,nbin=100,rel='DR4')

# For MzLS fill factors of 100% with a coverage of at least 1, 
# 99.5% with a coverage of at least 2, and 85% with a coverage of 3.
#FracExp=[0.005,0.145,0.85,0,0]

#print "MzLS depth histogram z-band"
#band='z'
#plotMaghist_pred(band,FracExp=FracExp,ndraw = 1e5,nbin=100,rel='DR4')

# --- run histogram deph peredictions

photometricReq('g',rel='DR3',survey='survename'):

