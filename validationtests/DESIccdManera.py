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
        extc = 3.303/2.751
    if band == 'r':
        extc = 2.285/2.751
    if band == 'z':
        extc = 1.263/2.751

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
    n = 0
    nl = []
    for i in range(0,len(f)):
        DS = 0
        year = int(f[i]['date_obs'].split('-')[0])
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


def depthfromIvarOld():
    
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
    #catalogue_name = 'DECaLS_DR2'+mjdw
    #catalogue_name = 'DECaLS_DR3'+mjdw
    catalogue_name = 'DECaLS_DR4'+mjdw
    #catalogue_name = '90prime_DR4'+mjdw
    #catalogue_name = 'MZLS_DR4'+mjdw
    pixoffset = 0 # How many pixels are being removed on the edge of each CCD? 15 for DES.
    #fname = inputdir+'ccds-annotated-decals.fits.gz'
    fname = inputdir+'ccds-annotated-dr4-90prime.fits.gz'
    #fname = inputdir+'ccds-annotated-dr4-mzls.fits.gz'
    
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
    indg = np.where((tbdata['filter'] == 'g') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
    #indr = np.where((tbdata['filter'] == 'r') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
    #indz = np.where((tbdata['filter'] == 'z') & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
    #sample_names = ['band_g', 'band_r', 'band_z']
    sample_names = ['band_g']
    #sample_names = ['band_r']
    #sample_names = ['band_z']
    #inds = [indg, indr, indz]
    inds = [indg]
    #inds = [indr]
    #inds = [indz]
    
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

    #DEBUGGING - MARC
    #print "len inds ", len(inds)
    #tbdata['ra0']=0
    #tbdata['ra1']=0
    #tbdata['ra2']=0
    #tbdata['ra3']=0
    #tbdata['dec0']=0
    #tbdata['dec1']=0
    #tbdata['dec2']=0
    #tbdata['dec3']=0
 
    # What properties to keep when reading the images? Should at least contain propertiesandoperations and the image corners.
    #propertiesToKeep = ['COADD_ID', 'ID', 'TILENAME', 'BAND', 'AIRMASS', 'SKYBRITE', 'SKYSIGMA', 'EXPTIME'] \
    #	+ ['FWHM', 'FWHM_MEAN', 'FWHM_PIXELFREE_MEAN', 'FWHM_FROMFLUXRADIUS_MEAN', 'NSTARS_ACCEPTED_MEAN'] \

    #previous
    #propertiesToKeep = [ 'filter', 'AIRMASS', 'FWHM','mjd_obs'] \
    # 	+ ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height'] \
    #    + ['ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3','ra','dec'] #, 'NAXIS1', 'NAXIS2']  \
    	 #\
    #    + ['PV1_'+str(i) for i in range(11)] + ['PV2_'+str(i) for i in range(11)] \
    #   + ['RALL', 'RAUL', 'RAUR', 'RALR', 'DECLL', 'DECUL', 'DECUR', 'DECLR', 'URALL', 'UDECLL', 'URAUR', 'UDECUR', 'COADD_RA', 'COADD_DEC'] \
    #    + ['COADD_MAGZP', 'MAGZP']

    # MARCM - no need for ra0 ra1 etc 
    propertiesToKeep = [ 'filter', 'AIRMASS', 'FWHM','mjd_obs'] \
    	+ ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height']
    
    # Create big table with all relevant properties. We will send it to the Quicksip library, which will do its magic.
    #tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [URAUL, UDECUL, URALR, UDECLR,ivar], names = propertiesToKeep + ['URAUL', 'UDECUL', 'URALR', 'UDECLR', 'ivar'])
    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar], names = propertiesToKeep + [ 'ivar'])
    
    # Do the magic! Read the table, create Healtree, project it into healpix maps, and write these maps.
    project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
    #project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside)
    
    # ------------------------------------------------------
    #plot depth maps
    # __________________
    
    depthminl = [21.5] #if doing multiple bands, add numbers there
    for i in range(0,len(sample_names)):
    	band = sample_names[i].split('_')[-1] 
    	depthmin = depthminl[i]
    	plotdepthfromIvar(band,depthmin=depthmin,mjdmax=mjdw,survey=catalogue_name)


def plotdepthfromIvar(band,depthmin=21.0,depthmax=24.0,mjdmax='',prop='ivar',op='total',survey='DECaLS_DR4',nside='1024',oversamp='1'):
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
	map = plt.scatter(ral,decl,c=magvalgmin, cmap=cm.rainbow,s=2., vmin=21.5, vmax=23.5, lw=0,edgecolors='none')
	cbar = plt.colorbar(map)
	#cbar.set_label(r'5$\sigma$ galaxy depth', rotation=270,labelpad=1)
	plt.xlabel('r.a. (degrees)')
	plt.ylabel('declination (degrees)')
	plt.title('Map of depth' +' for '+survey+' '+band+'-band')
	plt.xlim(0,360)
	plt.ylim(-30,90)
	#plt.show()
	plt.savefig(localdir+'depth_'+band+'_'+survey+str(nside)+'.png')
    	#plt.xscale('log')
	#pp.savefig()
	#pp.close()
	return True



def plotMaghist_proc_Nexp(band,Nexp,ndraw = 1e5,nbin=100,rel='DR3',survey='decals'):
	#this produces the histogram for Dustin's processed galaxy depth
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	if rel == 'DR2':
		f = fitsio.read(inputdir+'decals-ccds-annotated.fits')
	if rel == 'DR3':
		f = fitsio.read(inputdir+'ccds-annotated-'+survey+'.fits.gz')
        if rel == 'DR4':
            if (band == 'g' or band == 'r'):
                f = fitsio.read(inputdir+'ccds-annotated-dr4-90prime.fits.gz')
            if band == 'z' :
                f = fitsio.read(inputdir+'ccds-annotated-dr4-mzls.fits.gz') 
	
	nl = []
	nd = 0
	nbr = 0	
		
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

	n = 0
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			#if pid == '2014B-0404' or pid == '2013A-0741':
			DS = 1 #enforce 2015 data

		if f[i]['filter'] == band:
			
			if DS == 1:
				n += 1				
				if f[i]['dec'] > -20 and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True :   
				
				#if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
					#if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ext = f[i]['decam_extinction'][be]
					detsig1 = Magtonanomaggies(f[i]['galdepth'])/5. #total noise
					signalext = 1./10.**(-ext/2.5)
					nl.append(detsig1*signalext)
			
	print n	
	print len(nl)
	ng = len(nl)
	NTl = []
	for nd in range(0,int(ndraw)):
		detsigtoti = 0
		for i in range(0,Nexp):
			ind = int(random()*ng)
			detsig1 = nl[ind]
			detsigtoti += 1./detsig1**2.
		detsigtot = sqrt(1./detsigtoti)
		m = nanomaggiesToMag(detsigtot * 5.)
		if m > recm:
			nbr += 1.	
		NTl.append(m)
		n += 1.
	minN = min(NTl)
	maxN = max(NTl)+.0001
	print minN,maxN
	hl = zeros((nbin))
	for i in range(0,len(NTl)):
		bin = int(nbin*(NTl[i]-minN)/(maxN-minN))
		hl[bin] += 1
	Nl = []
	for i in range(0,len(hl)):
		Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))
	NTl = array(NTl)
	mean = sum(NTl)/float(len(NTl))
	std = sqrt(sum(NTl**2.)/float(len(NTl))-mean**2.)
	NTl.sort()
	if len(NTl)/2. != len(NTl)/2:
		med = NTl[len(NTl)/2+1]
	else:
		med = (NTl[len(NTl)/2+1]+NTl[len(NTl)/2])/2.
	print mean,med,std
	print 'percentage better than requirements '+str(nbr/float(nd))
	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages(localdir+'validationplots/'+rel+survey+band+str(Nexp)+'exposures.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title('MC '+str(Nexp)+' exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(nd))[:5]+'\n '+rel+' '+survey)
	#plt.xscale('log')
	pp.savefig()
	pp.close()
        print "----------"
	return True

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
#Nexp=3
#band='g'
#plotMaghist_proc_Nexp(band,Nexp,ndraw = 1e5,nbin=100,rel='DR3',survey='decals')
#
#Nexp=3
#band='r'
#plotMaghist_proc_Nexp(band,Nexp,ndraw = 1e5,nbin=100,rel='DR3',survey='decals')
#
#Nexp=3
#band='z'
#plotMaghist_proc_Nexp(band,Nexp,ndraw = 1e5,nbin=100,rel='DR3',survey='decals')

#Nexp=3
#band='g'
#plotMaghist_proc_Nexp(band,Nexp,ndraw = 1e5,nbin=100,rel='DR4',survey='90prime')

#Nexp=3
#band='r'
#plotMaghist_proc_Nexp(band,Nexp,ndraw = 1e5,nbin=100,rel='DR4',survey='90prime')

#Nexp=3
#band='z'
#plotMaghist_proc_Nexp(band,Nexp,ndraw = 1e5,nbin=100,rel='DR4',survey='MLZ')



