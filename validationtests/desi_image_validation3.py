#
import os
import numpy as np
import healpy as hp
import astropy.io.fits as pyfits
from multiprocessing import Pool
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from quicksipManera3 import *
import fitsio
from random import random
import matplotlib.colors as colors

### ------------ A couple of useful conversions -----------------------

def zeropointToScale(zp):
	return 10.**((zp - 22.5)/2.5)	

def nanomaggiesToMag(nm):
	return -2.5 * (log(nm,10.) - 9.)

def Magtonanomaggies(m):
	return 10.**(-m/2.5+9.)
	#-2.5 * (log(nm,10.) - 9.)

def thphi2radec(theta,phi):
        return 180./pi*phi,-(180./pi*theta-90)


# --------- Definitions for polinomial footprints -------

def convertCoordsToPoly (ra, dec) :
    poly = []
    for i in range(0,len(ra)) :
        poly.append((ra[i], dec[i]))
    return poly

def point_in_poly(x,y,poly):
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside

#### Class for plotting diverging colours to choosen mid point
# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


### ------------ SHARED CLASS: HARDCODED INPUTS GO HERE ------------------------
###    Please, add here your own harcoded values if any, so other may use them 

class mysample(object):
    """
    This class mantains the basic information of the sample
    to minimize hardcoded parameters in the test functions

    Everyone is meant to call mysample to obtain information like 
         - path to ccd-annotated files   : ccds
         - zero points                   : zp0
         - magnitude limits (recm)       : recm
         - photoz requirements           : phreq
         - extintion coefficient         : extc
         - extintion index               : be
         - mask var eqv. to blacklist_ok : maskname
         - predicted frac exposures      : FracExp   
         - footprint poly (in some cases) : polyFoot
    Current Inputs are: survey, DR, band, localdir) 
         survey: DECaLS, MZLS, BASS, DEShyb, NGCproxy
         DR:     DR3, DR4, DR5
         band:   g,r,z
         localdir: output directory
         (DEShyb is some DES proxy area; NGC is a proxy of North Gal Cap)
    """                                  


    def __init__(self,survey,DR,band,localdir,verb):
        """ 
        Initialize image survey, data release, band, output path
        Calculate variables and paths
        """   
        self.survey = survey
        self.DR     = DR
        self.band   = band
        self.localdir = localdir 
        self.verbose =verb
        # Check bands
        if(self.band != 'g' and self.band !='r' and self.band!='z'): 
            raise RuntimeError("Band seems wrong options are 'g' 'r' 'z'")        
              
        # Check surveys
        if(self.survey !='DECaLS' and  self.survey !='BASS' and self.survey !='MZLS' and self.survey !='DEShyb' and self.survey !='NGCproxy'):
            raise RuntimeError("Survey seems wrong options are 'DECAaLS' 'BASS' MZLS' ")

        # Annotated CCD paths  
        if(self.DR == 'DR3'):
            inputdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr3/'
            self.ccds =inputdir+'ccds-annotated-decals.fits.gz'
            self.catalog = 'DECaLS_DR3'
            if(self.survey != 'DECaLS'): raise RuntimeError("Survey name seems inconsistent; only DECaLS accepted")

        elif(self.DR == 'DR4'):
            inputdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr4/'
            if (band == 'g' or band == 'r'):
                #self.ccds = inputdir+'ccds-annotated-dr4-90prime.fits.gz'
                self.ccds = inputdir+'ccds-annotated-bass.fits.gz'
                self.catalog = 'BASS_DR4'
                if(self.survey != 'BASS'): raise RuntimeError("Survey name seems inconsistent")

            elif(band == 'z'):
                #self.ccds = inputdir+'ccds-annotated-dr4-mzls.fits.gz'
                self.ccds = inputdir+'ccds-annotated-mzls.fits.gz'
                self.catalog = 'MZLS_DR4'
                if(self.survey != 'MZLS'): raise RuntimeError("Survey name seems inconsistent")
            else: raise RuntimeError("Input sample band seems inconsisent")

        elif(self.DR == 'DR5'):
            inputdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr5/'
            self.ccds =inputdir+'ccds-annotated-dr5.fits.gz'
            if(self.survey == 'DECaLS') : 
                   self.catalog = 'DECaLS_DR5'
            elif(self.survey == 'DEShyb') :
                   self.catalog = 'DEShyb_DR5'
            elif(self.survey == 'NGCproxy' ) :
                   self.catalog = 'NGCproxy_DR5'
            else: raise RuntimeError("Survey name seems inconsistent")

        elif(self.DR == 'DR6'):
            inputdir = '/global/cscratch1/sd/dstn/dr6plus/'
            if (band == 'g'):
                self.ccds = inputdir+'ccds-annotated-90prime-g.fits.gz'
                self.catalog = 'BASS_DR6'
                if(self.survey != 'BASS'): raise RuntimeError("Survey name seems inconsistent")
            elif (band == 'r'):
                self.ccds = inputdir+'ccds-annotated-90prime-r.fits.gz'
                self.catalog = 'BASS_DR6'
                if(self.survey != 'BASS'): raise RuntimeError("Survey name seems inconsistent")
            elif(band == 'z'):
                self.ccds = inputdir+'ccds-annotated-mosaic-z.fits.gz'
                self.catalog = 'MZLS_DR6'
                if(self.survey != 'MZLS'): raise RuntimeError("Survey name seems inconsistent")
            else: raise RuntimeError("Input sample band seems inconsisent")

        elif(self.DR == 'DR7'):
            inputdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr7/'
            self.ccds =inputdir+'ccds-annotated-dr7.fits.gz'
            if(self.survey == 'DECaLS') : 
                   self.catalog = 'DECaLS_DR7'
            elif(self.survey == 'DEShyb') :
                   self.catalog = 'DEShyb_DR7'
            elif(self.survey == 'NGCproxy' ) :
                   self.catalog = 'NGCproxy_DR7'
            else: raise RuntimeError("Survey name seems inconsistent")


        else: raise RuntimeError("Data Realease seems wrong") 

        # Make directory for outputs if it doesn't exist
        dirplots = localdir + self.catalog
        try: 
            os.makedirs(dirplots)
            print("creating directory for plots", dirplots)
        except OSError:
            if not os.path.isdir(dirplots):
                raise
            
            
        # Predicted survey exposure fractions 
        if(self.survey =='DECaLS' or self.survey =='DEShyb' or self.survey =='NGCproxy'):
             # DECALS final survey will be covered by 
             # 1, 2, 3, 4, and 5 exposures in the following fractions: 
             self.FracExp=[0.02,0.24,0.50,0.22,0.02]
        elif(self.survey == 'BASS'):
             # BASS coverage fractions for 1,2,3,4,5 exposures are:
             self.FracExp=[0.0014,0.0586,0.8124,0.1203,0.0054,0.0019]
        elif(self.survey == 'MZLS'):
             # For MzLS fill factors of 100% with a coverage of at least 1, 
             # 99.5% with a coverage of at least 2, and 85% with a coverage of 3.
             self.FracExp=[0.005,0.145,0.85,0,0]
        else:
             raise RuntimeError("Survey seems to have wrong options for fraction of exposures ")


        #Bands inputs
        if band == 'g':
            self.be = 1
            self.extc = 3.303  #/2.751
            self.zp0 = 25.08
            self.recm = 24.
            self.phreq = 0.01
        if band == 'r':
            self.be = 2
            self.extc = 2.285  #/2.751
            self.zp0 = 25.29
            self.recm = 23.4
            self.phreq = 0.01
        if band == 'z':
            self.be = 4
            self.extc = 1.263  #/2.751
            self.zp0 = 24.92
            self.recm = 22.5
            self.phreq = 0.02

# ------------------------------------------------------------------
# --- Footprints ----


polyDEScoord=np.loadtxt('/global/homes/m/manera/round13-poly-radec.dat')
polyDES = convertCoordsToPoly(polyDEScoord[:,0], polyDEScoord[:,1])
        
def InDEShybFootprint(RA,DEC):
    ''' Decides if it is in DES footprint  '''
    # The DES footprint 
    if(RA > 180) : RA = RA -360
    return point_in_poly(RA, DEC, polyDES) 

def InNGCproxyFootprint(RA):
    if(RA < -180 ) : RA = RA + 360
    if( 100 <= RA <= 300 ) : return True 
    return False 


def plot_magdepth2D(sample,ral,decl,depth,mapfile,mytitle):
    # Plot depth 
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    ralB = [ ra-360 if ra > 300 else ra for ra in ral ]
    vmax = sample.recm + 2.0 
    vmin = sample.recm - 2.0
    #mapa = plt.scatter(ralB,decl,c=depth, cmap=cm.gnuplot,s=2., vmin=vmin, vmax=vmax, lw=0,edgecolors='none')
    mapa = plt.scatter(ralB,decl,c=depth, cmap=cm.RdYlBu_r,s=2., vmin=vmin, vmax=vmax, lw=0,edgecolors='none')
    mapa.cmap.set_over('lawngreen')
    cbar = plt.colorbar(mapa,extend='both')
    plt.xlabel('r.a. (degrees)')
    plt.ylabel('declination (degrees)')
    plt.title(mytitle)
    plt.xlim(-60,300)
    plt.ylim(-30,90)
    print('saving plot image to %s', mapfile)
    plt.savefig(mapfile)
    plt.close()
    return



def plot_magdepth2Db(sample,ral,decl,depth,mapfile,mytitle):
    # Plot depth 
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    #ralB = [ ra-360 if ra > 300 else ra for ra in ral ]
    #vmax = sample.recm
    #vmin = vmax - 2.0

   # test
   # plt.imshow(ras, cmap=cmap, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
   # test
    #vmaxext=vmax+2.0 
#    mapa = plt.scatter(ralB,decl,c=depth, cmap=cm.gnuplot,s=2., norm=MidpointNormalize(midpoint=vmax,vmin=vmin,vmax=vmaxext), vmin=vmin, vmax=vmaxext, lw=0,edgecolors='none')
    #mapa = plt.scatter(ralB,decl,c=depth, cmap=cm.gnuplot,s=2., vmin=vmin, vmax=vmax, lw=0,edgecolors='none')
    #mapa = plt.scatter(ralB,decl,c=depth, cmap=cm.gnuplot,s=2., vmin=vmin, vmax=vmax, lw=0,edgecolors='none')
    mapa = plt.scatter(ral,decl,c=depth, cmap=cm.gnuplot,s=2,vmax=23.0)
    mapa.cmap.set_over('lawngreen')
    cbar = plt.colorbar(mapa,extend='both')
    #plt.xlabel('r.a. (degrees)')
    #plt.ylabel('declination (degrees)')
    #plt.title(mytitle)
    #plt.xlim(-60,300)
    #plt.ylim(-30,90)
    print('saving plot to ', mapfile)
    plt.savefig(mapfile)
    plt.close()
    return 


def prova(sample):
    magaux = [20.0+i*0.005 for i in range(1001)]
    xaux = [100.0 + float(i%100) for i in range(1001)]
    yaux = [100.0 + float(i)/100 for i in range(1001)]
    mytitle = 'prova'
    mapfile = '/global/homes/m/manera/DESI/validation-outputs/prova.png'
    plot_magdepth2Db(sample,xaux,yaux,magaux,mapfile,mytitle)
    #plot_magdepth2Db(sample,ral,decl,depth,mapfile,mytitle):
# ------------------------------------------------------------------
# ------------ VALIDATION TESTS ------------------------------------
# ------------------------------------------------------------------
# Note: part of the name of the function should startw with number valXpX 

def val3p4c_depthfromIvarOLD(sample):    
    """
       Requirement V3.4
       90% filled to g=24, r=23.4 and z=22.5 and 95% and 98% at 0.3/0.6 mag shallower.

       Produces extinction correction magnitude maps for visual inspection

       MARCM stable version, improved from AJR quick hack 
       This now included extinction from the exposures
       Uses quicksip subroutines from Boris, corrected 
       for a bug I found for BASS and MzLS ccd orientation
    """
    nside = 1024       # Resolution of output maps
    nsideSTR='1024'    # same as nside but in string format
    nsidesout = None   # if you want full sky degraded maps to be written
    ratiores = 1       # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
    mode = 1           # 1: fully sequential, 2: parallel then sequential, 3: fully parallel
    pixoffset = 0      # How many pixels are being removed on the edge of each CCD? 15 for DES.
    oversamp='1'       # ratiores in string format
    
    band = sample.band
    catalogue_name = sample.catalog
    fname = sample.ccds    
    localdir = sample.localdir
    extc = sample.extc

    #Read ccd file 
    tbdata = pyfits.open(fname)[1].data
    
    # ------------------------------------------------------
    # Obtain indices
    auxstr='band_'+band
    sample_names = [auxstr]
    if(sample.DR == 'DR3'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
    elif(sample.DR == 'DR4'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['bitmask'] == 0)) 
    elif(sample.DR == 'DR5'):
        if(sample.survey == 'DECaLS'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
        elif(sample.survey == 'DEShyb'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True) & (list(map(InDEShybFootprint,tbdata['ra'],tbdata['dec']))))
        elif(sample.survey == 'NGCproxy'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True) & (list(map(InNGCproxyFootprint,tbdata['ra'])))) 
    elif(sample.DR == 'DR6'):
        inds = np.where((tbdata['filter'] == band)) 
    elif(sample.DR == 'DR7'):
        if(sample.survey == 'DECaLS'):
            inds = np.where((tbdata['filter'] == band)) 
        elif(sample.survey == 'DEShyb'):
            inds = np.where((tbdata['filter'] == band) & (list(map(InDEShybFootprint,tbdata['ra'],tbdata['dec']))))
        elif(sample.survey == 'NGCproxy'):
            inds = np.where((tbdata['filter'] == band) & (list(map(InNGCproxyFootprint,tbdata['ra'])))) 

    #Read data 
    #obtain invnoisesq here, including extinction 
    nmag = Magtonanomaggies(tbdata['galdepth']-extc*tbdata['EBV'])/5.
    ivar= 1./nmag**2.

    hits=np.ones(np.shape(ivar)) 

    # What properties do you want mapped?
    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    propertiesandoperations = [ ('ivar', '', 'total'), ('hits','','total') ]

 
    # What properties to keep when reading the images? 
    # Should at least contain propertiesandoperations and the image corners.
    # MARCM - actually no need for ra dec image corners.   
    # Only needs ra0 ra1 ra2 ra3 dec0 dec1 dec2 dec3 only if fast track appropriate quicksip subroutines were implemented 
    #propertiesToKeep = [ 'filter', 'FWHM','mjd_obs'] \
    # 	+ ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height']
    propertiesToKeep = [ 'filter', 'FWHM','mjd_obs'] \
    	+ ['RA', 'DEC', 'ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3']
    
    # Create big table with all relevant properties. 

    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar] + [hits], names = propertiesToKeep + [ 'ivar', 'hits'])
    
    # Read the table, create Healtree, project it into healpix maps, and write these maps.
    # Done with Quicksip library, note it has quite a few hardcoded values (use new version by MARCM for BASS and MzLS) 
    # project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside)
    #project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, localdir, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
    project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, localdir, sample_names, inds, nside)
 
    # Read Haelpix maps from quicksip  
    prop='ivar'
    op='total'
    vmin=21.0
    vmax=24.0

    fname2=localdir+catalogue_name+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+\
    catalogue_name+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
    f = fitsio.read(fname2)

    # HEALPIX DEPTH MAPS 
    # convert ivar to depth 
    import healpy as hp
    from healpix3 import pix2ang_ring,thphi2radec

    ral = []
    decl = []
    val = f['SIGNAL']
    pix = f['PIXEL']
 
    # Obtain values to plot 
    if (prop == 'ivar'):
        myval = []
        mylabel='depth' 
            
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

    npix=len(pix)

    print('Area is ', npix/(float(nside)**2.*12)*360*360./pi, ' sq. deg.')
    print(below, 'of ', npix, ' pixels are not plotted as their ', mylabel,' < ', vmin)
    print('Within the plot, min ', mylabel, '= ', min(myval), ' and max ', mylabel, ' = ', max(myval))


    # Plot depth 
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm

    mapa = plt.scatter(ral,decl,c=myval, cmap=cm.rainbow,s=2., vmin=vmin, vmax=vmax, lw=0,edgecolors='none')
    cbar = plt.colorbar(mapa)
    plt.xlabel('r.a. (degrees)')
    plt.ylabel('declination (degrees)')
    plt.title('Map of '+ mylabel +' for '+catalogue_name+' '+band+'-band')
    plt.xlim(0,360)
    plt.ylim(-30,90)
    mapfile=localdir+mylabel+'_'+band+'_'+catalogue_name+str(nside)+'.png'
    print('saving plot to ', mapfile)
    plt.savefig(mapfile)
    plt.close()
    #plt.show()
    #cbar.set_label(r'5$\sigma$ galaxy depth', rotation=270,labelpad=1)
    #plt.xscale('log')


    # Statistics depths

    deptharr=np.array(myval)
    p90=np.percentile(deptharr,10)
    p95=np.percentile(deptharr,5)
    p98=np.percentile(deptharr,2)
    med=np.percentile(deptharr,50)
    mean = sum(deptharr)/float(np.size(deptharr))  # 1M array, too long for precision
    std = sqrt(sum(deptharr**2.)/float(len(deptharr))-mean**2.)

   

    ndrawn=np.size(deptharr)
    print("Total pixels", np.size(deptharr), "probably too many for exact mean and std")
    print("Mean = ", mean, "; Median = ", med ,"; Std = ", std)
    print("Results for 90% 95% and 98% are: ", p90, p95, p98)

    # Statistics pases
    prop = 'hits'
    op = 'total'
    fname2=localdir+catalogue_name+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+\
    catalogue_name+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
    f = fitsio.read(fname2)
    
    hitsb=f['SIGNAL']
    hist, bin_edges =np.histogram(hitsb,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,100],density=True)
    #print hitsb[1000:10015]
    #hist, bin_edges =np.histogram(hitsb,density=True)
    print("Percentage of hits for 0,1,2., to >7 pases\n", end=' ') 
    #print bin_edges 
    print(hist)
    #print 100*hist

    return mapfile 


def val3p4c_depthfromIvar(sample,Nexpmin=1):    
    """
       Requirement V3.4
       90% filled to g=24, r=23.4 and z=22.5 and 95% and 98% at 0.3/0.6 mag shallower.

       Produces extinction correction magnitude maps for visual inspection
       MARCM includes min number of exposures
       MARCM stable version, improved from AJR quick hack 
       This now included extinction from the exposures
       Uses quicksip subroutines from Boris, corrected 
       for a bug I found for BASS and MzLS ccd orientation
    """
    nside = 1024       # Resolution of output maps
    nsideSTR='1024'    # same as nside but in string format
    nsidesout = None   # if you want full sky degraded maps to be written
    ratiores = 1       # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
    mode = 1           # 1: fully sequential, 2: parallel then sequential, 3: fully parallel
    pixoffset = 0      # How many pixels are being removed on the edge of each CCD? 15 for DES.
    oversamp='1'       # ratiores in string format
    
    band = sample.band
    catalogue_name = sample.catalog
    fname = sample.ccds    
    localdir = sample.localdir
    extc = sample.extc

    #Read ccd file 
    tbdata = pyfits.open(fname)[1].data
    
    # ------------------------------------------------------
    # Obtain indices
    auxstr='band_'+band
    sample_names = [auxstr]
    if(sample.DR == 'DR3'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
    elif(sample.DR == 'DR4'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['bitmask'] == 0)) 
    elif(sample.DR == 'DR5'):
        if(sample.survey == 'DECaLS'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
        elif(sample.survey == 'DEShyb'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True) & (list(map(InDEShybFootprint,tbdata['ra'],tbdata['dec']))))
        elif(sample.survey == 'NGCproxy'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True) & (list(map(InNGCproxyFootprint,tbdata['ra'])))) 
    elif(sample.DR == 'DR6'):
        inds = np.where((tbdata['filter'] == band)) 
    elif(sample.DR == 'DR7'):
        if(sample.survey == 'DECaLS'):
            inds = np.where((tbdata['filter'] == band)) 
        elif(sample.survey == 'DEShyb'):
            inds = np.where((tbdata['filter'] == band) & (list(map(InDEShybFootprint,tbdata['ra'],tbdata['dec']))))
        elif(sample.survey == 'NGCproxy'):
            inds = np.where((tbdata['filter'] == band) & (list(map(InNGCproxyFootprint,tbdata['ra'])))) 


    #Read data 
    #obtain invnoisesq here, including extinction 
    nmag = Magtonanomaggies(tbdata['galdepth']-extc*tbdata['EBV'])/5.
    ivar= 1./nmag**2.

    hits=np.ones(np.shape(ivar)) 

    # What properties do you want mapped?
    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    propertiesandoperations = [ ('ivar', '', 'total'), ('hits','','total') ]

 
    # What properties to keep when reading the images? 
    # Should at least contain propertiesandoperations and the image corners.
    # MARCM - actually no need for ra dec image corners.   
    # Only needs ra0 ra1 ra2 ra3 dec0 dec1 dec2 dec3 only if fast track appropriate quicksip subroutines were implemented 
    #propertiesToKeep = [ 'filter', 'FWHM','mjd_obs'] \
    # 	+ ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height']
    propertiesToKeep = [ 'filter', 'FWHM','mjd_obs'] \
    	+ ['RA', 'DEC', 'ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3']
    
    # Create big table with all relevant properties. 

    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar] + [hits], names = propertiesToKeep + [ 'ivar', 'hits'])
    
    # Read the table, create Healtree, project it into healpix maps, and write these maps.
    # Done with Quicksip library, note it has quite a few hardcoded values (use new version by MARCM for BASS and MzLS) 
    # project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside)
    #project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, localdir, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
    project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, localdir, sample_names, inds, nside)
 
    # Read Haelpix maps from quicksip  
    prop='ivar'
    op='total'
    vmin=21.0
    vmax=24.0

    fname2=localdir+catalogue_name+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+\
    catalogue_name+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
    f = fitsio.read(fname2)

    # HEALPIX DEPTH MAPS 
    # convert ivar to depth 
    import healpy as hp
    from healpix3 import pix2ang_ring,thphi2radec

    ral = []
    decl = []
    val = f['SIGNAL']
    pix = f['PIXEL']

    #get hits
    prop = 'hits'
    op = 'total'
    fname2=localdir+catalogue_name+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+\
    catalogue_name+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
    f = fitsio.read(fname2)
    hitsb=f['SIGNAL']
 
    # Obtain values to plot 
    #if (prop == 'ivar'):
    myval = []
    mylabel='depth' 
            
    below=0 
    for i in range(0,len(val)):
       depth=nanomaggiesToMag(sqrt(1./val[i]) * 5.)
       npases=hitsb[i]
       if(npases >= Nexpmin  ):
           myval.append(depth)
           th,phi = hp.pix2ang(int(nside),pix[i])
           ra,dec = thphi2radec(th,phi)
           ral.append(ra)
           decl.append(dec)
           if(depth < vmin):
              below=below+1

    npix=len(myval)

    print('Area is ', npix/(float(nside)**2.*12)*360*360./pi, ' sq. deg.')
    print(below, 'of ', npix, ' pixels are not plotted as their ', mylabel,' < ', vmin)
    print('Within the plot, min ', mylabel, '= ', min(myval), ' and max ', mylabel, ' = ', max(myval))


    # Plot depth 
    #from matplotlib import pyplot as plt
    #import matplotlib.cm as cm

    #ralB = [ ra-360 if ra > 300 else ra for ra in ral ]
    #vmax = sample.recm
    #vmin = vmax - 2.0
      
    #mapa = plt.scatter(ralB,decl,c=myval, cmap=cm.rainbow,s=2., vmin=vmin, vmax=vmax, lw=0,edgecolors='none')
    #mapa = plt.scatter(ralB,decl,c=myval, cmap=cm.gnuplot,s=2., vmin=vmin, vmax=vmax, lw=0,edgecolors='none')
    #mapa.cmap.set_over('yellowgreen')
    #cbar = plt.colorbar(mapa,extend='both')
    #plt.xlabel('r.a. (degrees)')
    #plt.ylabel('declination (degrees)')
    #plt.title('Map of '+ mylabel +' for '+catalogue_name+' '+band+'-band \n with 3 or more exposures')
    #plt.xlim(-60,300)
    #plt.ylim(-30,90)
    #mapfile=localdir+mylabel+'_'+band+'_'+catalogue_name+str(nside)+'.png'
    #print 'saving plot to ', mapfile
    #plt.savefig(mapfile)
    #plt.close()
    #plt.show()
    #cbar.set_label(r'5$\sigma$ galaxy depth', rotation=270,labelpad=1)
    #plt.xscale('log')
    mapfile=localdir+mylabel+'_'+band+'_'+catalogue_name+str(nside)+'.png'
    mytitle='Map of '+ mylabel +' for '+catalogue_name+' '+band+'-band \n with '+str(Nexpmin)+\
           ' or more exposures'
    plot_magdepth2D(sample,ral,decl,myval,mapfile,mytitle)


    # Statistics depths

    deptharr=np.array(myval)
    p90=np.percentile(deptharr,10)
    p95=np.percentile(deptharr,5)
    p98=np.percentile(deptharr,2)
    med=np.percentile(deptharr,50)
    mean = sum(deptharr)/float(np.size(deptharr))  # 1M array, too long for precision
    std = sqrt(sum(deptharr**2.)/float(len(deptharr))-mean**2.)

   

    ndrawn=np.size(deptharr)
    print("Total pixels", np.size(deptharr), "probably too many for exact mean and std")
    print("Mean = ", mean, "; Median = ", med ,"; Std = ", std)
    print("Results for 90% 95% and 98% are: ", p90, p95, p98)

    # Statistics pases
    #prop = 'hits'
    #op = 'total'
    #fname2=localdir+catalogue_name+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+\
    #catalogue_name+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
    #f = fitsio.read(fname2)
    
    #hitsb=f['SIGNAL']
    hist, bin_edges =np.histogram(hitsb,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,100],density=True)
    #print hitsb[1000:10015]
    #hist, bin_edges =np.histogram(hitsb,density=True)
    print("Percentage of hits for 0,1,2., to >7 pases\n", end=' ') 
    #print bin_edges 
    print(hist)
    #print 100*hist

    return mapfile 


#def plot_magdepth2D(sample,ral,decl,depth,mapfile,mytitle):
#    # Plot depth 
#    from matplotlib import pyplot as plt
#    import matplotlib.cm as cm
#    ralB = [ ra-360 if ra > 300 else ra for ra in ral ]
#    vmax = sample.recm
#    vmin = vmax - 2.0
#    #test
#    vmaxup= vmax+2.0
#    mapa = plt.scatter(ralB,decl,c=depth, cmap=cm.gnuplot,s=2., norm=MidpointNormalize(midpoint=vmax,vmin=vmin, vmax=vmaxup), vmin=vmin, vmax=vmaxup, lw=0,edgecolors='none')
 #   mapa = plt.scatter(ralB,decl,c=depth, cmap=cm.gnuplot,s=2., vmin=vmin, vmax=vmax, lw=0,edgecolors='none')
#    mapa.cmap.set_over('yellowgreen')
#    cbar = plt.colorbar(mapa,extend='both')
#    plt.xlabel('r.a. (degrees)')
#    plt.ylabel('declination (degrees)')
#    plt.title(mytitle)
#    plt.xlim(-60,300)
#    plt.ylim(-30,90)
#    print('saving plot to ', mapfile)
#    plt.savefig(mapfile)
#    plt.close()
#    return mapfile

def val3p4b_maghist_pred(sample,ndraw=1e5, nbin=100, vmin=21.0, vmax=25.0, Nexpmin=1):    
    """
       Requirement V3.4
       90% filled to g=24, r=23.4 and z=22.5 and 95% and 98% at 0.3/0.6 mag shallower.

       MARCM 
       Makes histogram of predicted magnitudes 
       by MonteCarlo from exposures converving fraction of number of exposures
       This produces the histogram for Dustin's processed galaxy depth
    """


    # Check fraction of number of exposures adds to 1. 
    if( abs(sum(sample.FracExp) - 1.0) > 1e-5 ):
       raise ValueError("Fration of number of exposures don't add to one")

    # Survey inputs
    rel = sample.DR
    catalogue_name = sample.catalog
    band = sample.band
    be = sample.be
    zp0 = sample.zp0
    recm = sample.recm
    verbose = sample.verbose

    f = fitsio.read(sample.ccds)
    
    #read in magnitudes including extinction
    counts2014 = 0
    counts20 = 0
    nl = []
    for i in range(0,len(f)):
        #year = int(f[i]['date_obs'].split('-')[0])
        #if (year <= 2014): counts2014 = counts2014 + 1
        if f[i]['dec'] < -20 : counts20 = counts20 + 1


        if(sample.DR == 'DR3'): 

            if f[i]['filter'] == sample.band and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True :   

                magext = f[i]['galdepth'] - f[i]['decam_extinction'][be]
                nmag = Magtonanomaggies(magext)/5. #total noise
                nl.append(nmag)


        if(sample.DR == 'DR4'): 
             if f[i]['filter'] == sample.band and f[i]['photometric'] == True and f[i]['bitmask'] == 0 :   

                 magext = f[i]['galdepth'] - f[i]['decam_extinction'][be]
                 nmag = Magtonanomaggies(magext)/5. #total noise
                 nl.append(nmag)

        if(sample.DR == 'DR5'): 
             if f[i]['filter'] == sample.band and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True :   

                 if(sample.survey == 'DEShyb'):
                    RA = f[i]['ra']
                    DEC = f[i]['dec'] # more efficient to put this inside the function directly
                    if(not InDEShybFootprint(RA,DEC) ): continue   # skip if not in DEShyb

                 if(sample.survey == 'NGCproxy'):
                    RA = f[i]['ra'] # more efficient to put this inside the function directly
                    if(not InNGCproxyFootprint(RA) ): continue   # skip if not in NGC 

                 magext = f[i]['galdepth'] - f[i]['decam_extinction'][be]
                 nmag = Magtonanomaggies(magext)/5. #total noise
                 nl.append(nmag)

        if(sample.DR == 'DR6'): 
             if f[i]['filter'] == sample.band :   

                 magext = f[i]['galdepth'] - f[i]['decam_extinction'][be]
                 nmag = Magtonanomaggies(magext)/5. #total noise
                 nl.append(nmag)

        if(sample.DR == 'DR7'): 
             #if f[i]['filter'] == sample.band:   
             if(sample.band == 'r'): band_aux = b'r'
             if(sample.band == 'g'): band_aux = b'g'
             if(sample.band == 'z'): band_aux = b'z'
             if f[i]['filter'] == band_aux:   

                 if(sample.survey == 'DEShyb'):
                    RA = f[i]['ra']
                    DEC = f[i]['dec'] # more efficient to put this inside the function directly
                    if(not InDEShybFootprint(RA,DEC) ): continue   # skip if not in DEShyb

                 if(sample.survey == 'NGCproxy'):
                    RA = f[i]['ra'] # more efficient to put this inside the function directly
                    if(not InNGCproxyFootprint(RA) ): continue   # skip if not in NGC 

                 magext = f[i]['galdepth'] - f[i]['decam_extinction'][be]
                 nmag = Magtonanomaggies(magext)/5. #total noise
                 nl.append(nmag)
                
    ng = len(nl)
    print("-----------")
    if(verbose) : print("Number of objects = ", len(f))
    #if(verbose) : print "Counts before or during 2014 = ", counts2014
    if(verbose) : print("Counts with dec < -20 = ", counts20)
    print("Number of objects in the sample = ", ng) 

    #Monte Carlo to predict magnitudes histogram 
    ndrawn = 0
    nbr = 0	
    NTl = []
    n = 0
    for indx, f in enumerate(sample.FracExp,1) : 
        Nexp = indx # indx starts at 1 bc argument on enumearate :-), thus is the number of exposures
        nd = int(round(ndraw * f))
        if(Nexp < Nexpmin): nd=0 
        ndrawn=ndrawn+nd

        for i in range(0,nd):

            detsigtoti = 0
            for j in range(0,Nexp):
                ind = int(random()*ng)
                #if(ind >= len(nl)): print("%d %d %d",ind,len(nl),ng)
                #if(ind < 0 ): print("%d %d %d",ind,len(nl),ng)
                #print("%d",ind)
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
    p90=np.percentile(NTl,10)
    p95=np.percentile(NTl,5)
    p98=np.percentile(NTl,2)
    mean = sum(NTl)/float(len(NTl))
    std = sqrt(sum(NTl**2.)/float(len(NTl))-mean**2.)

    NTl.sort()
    #if len(NTl) /2. != len(NTl)/2:
    if len(NTl) % 2 != 0:
        med = NTl[len(NTl)//2+1]
    else:
        med = (NTl[len(NTl)//2+1]+NTl[len(NTl)//2])/2.

    print("Total images drawn with either ", Nexpmin, " to 5 exposures", ndrawn)
    print("We are in fact runing with a minimum of", Nexpmin, "exposures")
    print("Mean = ", mean, "; Median = ", med ,"; Std = ", std)
    print('percentage better than requirements = '+str(nbr/float(ndrawn)))
    if(sample.band =='g'): print("Requirements are > 90%, 95% and 98% at 24, 23.7, 23.4")
    if(sample.band =='r'): print("Requirements are > 90%, 95% and 98% at 23.4, 23.1, 22.8")
    if(sample.band =='z'): print("Requirements are > 90%, 95% and 98% at 22.5, 22.2, 21.9")
    print("Results are: ", p90, p95, p98)

    # Prepare historgram 
    minN = max(min(NTl),vmin)
    maxN = max(NTl)+.0001
    hl = np.zeros((nbin)) # histogram counts
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
    NTl = np.array(NTl)

    print("min,max depth = ",min(NTl), max(NTl)) 
    print("counts below ", minN, " = ", lowcounts)


    #### Ploting histogram 
    fname=sample.localdir+'validationplots/'+sample.catalog+sample.band+'_pred_exposures.png'
    print("saving histogram plot in", fname) 

       #--- pdf version --- 
       #from matplotlib.backends.backend_pdf import PdfPages
       #pp = PdfPages(fname)	

    plt.clf()
    plt.plot(Nl,hl,'k-')
    plt.xlabel(r'5$\sigma$ '+sample.band+ ' depth')
    plt.ylabel('# of images')
    plt.title('MC combined exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(ndrawn))[:5]+'\n '+catalogue_name)
    #plt.xscale('log')     # --- pdf --- 
    plt.savefig(fname)       #pp.savefig()
    plt.close              #pp.close()
    return fname 


def v5p1e_photometricReqPlot(sample):
    """
    No region > 3deg will be based upon non-photometric observations

    Produces a plot of zero-point variations across the survey to inspect visually    
    MARCM It can be extended to check the 3deg later if is necessary 
    """

    nside = 1024       # Resolution of output maps
    nsideSTR = '1024'  # String value for nside
    nsidesout = None   # if you want full sky degraded maps to be written
    ratiores = 1       # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
    oversamp = '1'      # string oversaple value
    mode = 1           # 1: fully sequential, 2: parallel then sequential, 3: fully parallel
    
    pixoffset = 0      # How many pixels are being removed on the edge of each CCD
    
    mjd_max = 10e10
    mjdw = ''
    rel = sample.DR
    catalogue_name = sample.catalog
    band = sample.band 
    sample_names = ['band_'+band]
    localdir = sample.localdir
    verbose = sample.verbose 
   
    tbdata = pyfits.open(sample.ccds)[1].data
    
    if(sample.DR == 'DR3'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['blacklist_ok'] == True)) 
    if(sample.DR == 'DR4'):    
        inds = np.where((tbdata['filter'] == band) & (tbdata['bitmask'] == 0)) 
    elif(sample.DR == 'DR5'):
        if(sample.survey == 'DECaLS'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['blacklist_ok'] == True)) 
        elif(sample.survey == 'DEShyb'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['blacklist_ok'] == True) & (list(map(InDEShybFootprint,tbdata['ra'],tbdata['dec']))))
        elif(sample.survey == 'NGCproxy'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['blacklist_ok'] == True) & (list(map(InNGCproxyFootprint,tbdata['ra'])))) 
    if(sample.DR == 'DR6'):    
        inds = np.where((tbdata['filter'] == band)) 
    elif(sample.DR == 'DR7'):
        if(sample.survey == 'DECaLS'):
            inds = np.where((tbdata['filter'] == band)) 
        elif(sample.survey == 'DEShyb'):
            inds = np.where((tbdata['filter'] == band) & (list(map(InDEShybFootprint,tbdata['ra'],tbdata['dec']))))
        elif(sample.survey == 'NGCproxy'):
            inds = np.where((tbdata['filter'] == band) & (list(map(InNGCproxyFootprint,tbdata['ra'])))) 


    #Read data 
    #obtain invnoisesq here, including extinction 
    zptvar = tbdata['CCDPHRMS']**2/tbdata['CCDNMATCH']
    zptivar = 1./zptvar
    nccd = np.ones(len(tbdata))

    # What properties do you want mapped?
    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    quicksipVerbose(sample.verbose)
    propertiesandoperations = [('zptvar','','min')]
    #propertiesandoperations = [ ('zptvar', '', 'total') , ('zptvar','','min') , ('nccd','','total') , ('zptivar','','total')]

 
    # What properties to keep when reading the images? 
    #Should at least contain propertiesandoperations and the image corners.
    # MARCM - actually no need for ra dec image corners.   
    # Only needs ra0 ra1 ra2 ra3 dec0 dec1 dec2 dec3 only if fast track appropriate quicksip subroutines were implemented 
    propertiesToKeep = [ 'filter', 'AIRMASS', 'FWHM','mjd_obs'] \
    	+ ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height']
    
    # Create big table with all relevant properties. 
    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [zptvar,zptivar,nccd], names = propertiesToKeep + [ 'zptvar','zptivar','nccd'])
    
    # Read the table, create Healtree, project it into healpix maps, and write these maps.
    # Done with Quicksip library, note it has quite a few hardcoded values (use new version by MARCM for BASS and MzLS) 
    # project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside)
    project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, localdir, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
 
    # ----- Read healpix maps for first case [zptvar min]-------------------------
    prop='zptvar'
    op= 'min'


    fname=localdir+catalogue_name+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+catalogue_name+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
    f = fitsio.read(fname)

    ral = []
    decl = []
    val = f['SIGNAL']
    pix = f['PIXEL']

    # -------------- plot of values ------------------
    print('Plotting min zpt rms')
    myval = []
    for i in range(0,len(val)):
             
        myval.append(1.086 * np.sqrt(val[i])) #1.086 converts d(mag) into d(flux) 
        th,phi = hp.pix2ang(int(nside),pix[i])
        ra,dec = thphi2radec(th,phi)
        ral.append(ra)
        decl.append(dec)
      
    mylabel = 'min-zpt-rms-flux'
    vmin = 0.0 #min(myval)
    vmax = 0.03 #max(myval) 
    npix = len(myval)
    below = 0
    print('Min and Max values of ', mylabel, ' values is ', min(myval), max(myval))
    print('Number of pixels is ', npix)
    print('Area is ', npix/(float(nside)**2.*12)*360*360./pi, ' sq. deg.')


    import matplotlib.cm as cm
    mapa = plt.scatter(ral,decl,c=myval, cmap=cm.rainbow,s=2., vmin=vmin, vmax=vmax, lw=0,edgecolors='none')
    
    fname = localdir+mylabel+'_'+band+'_'+catalogue_name+str(nside)+'.png'
    cbar = plt.colorbar(mapa)
    plt.xlabel('r.a. (degrees)')
    plt.ylabel('declination (degrees)')
    plt.title('Map of '+ mylabel +' for '+catalogue_name+' '+band+'-band')
    plt.xlim(0,360)
    plt.ylim(-30,90)
    plt.savefig(fname)
    plt.close()


    #-plot of status in udgrade maps:  nside64 = 1.406 deg pix size 

    phreq = sample.phreq 

    # Obtain values to plot

    nside2 = 64  # 1.40625 deg per pixel
    npix2 = hp.nside2npix(nside2)
    myreq = np.zeros(npix2)  # 0 off footprint, 1 at least one pass requirement, -1 none pass requirement
    ral = np.zeros(npix2)
    decl = np.zeros(npix2)
    mylabel = 'photometric-pixels'
    if(verbose) : print('Plotting photometric requirement')

    for i in range(0,len(val)):
        th,phi = hp.pix2ang(int(nside),pix[i])
        ipix = hp.ang2pix(nside2,th,phi)
        dF= 1.086 * (sqrt(val[i]))   # 1.086 converts d(magnitudes) into d(flux)

        if(dF < phreq):
             myreq[ipix]=1
        else:
             if(myreq[ipix] == 0): myreq[ipix]=-1

    below=sum( x for x in myreq if x < 0) 
    below=-below
    print('Number of udegrade pixels with ', mylabel,' > ', phreq, ' for all subpixels =', below)
    print('nside of udgraded pixels is : ', nside2)

    return fname 


def v3p5_Areas(sample1,sample2):
    """
    450 sq deg overlap between all instruments (R3.5) 
    14000 sq deg filled  (R3.1) 

    MARCM
    Uses healpix pixels projected from ccd to count standing, join and overlap areas 
    Produces a map of the overlap     
    It can be extended to 3 samples if necessary 
    """

    nside = 1024       # Resolution of output maps
    nsideSTR = '1024'  # String value for nside
    nsidesout = None   # if you want full sky degraded maps to be written
    ratiores = 1       # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
    oversamp = '1'      # string oversaple value
    mode = 1           # 1: fully sequential, 2: parallel then sequential, 3: fully parallel
    
    pixoffset = 0      # How many pixels are being removed on the edge of each CCD
    mjdw = ''
 
    # ----- sample1 ------ 
    tbdata = pyfits.open(sample1.ccds)[1].data
    if(sample1.DR == 'DR3'):
        inds = np.where((tbdata['filter'] == sample1.band) & (tbdata['blacklist_ok'] == True)) 
    if(sample1.DR == 'DR4'):    
        inds = np.where((tbdata['filter'] == sample1.band) & (tbdata['bitmask'] == 0)) 
    elif(sample.DR == 'DR5'):
        if(sample.survey == 'DECaLS'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['blacklist_ok'] == True)) 
        elif(sample.survey == 'DEShyb'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['blacklist_ok'] == True) & (list(map(InDEShybFootprint,tbdata['ra'],tbdata['dec']))))
        elif(sample.survey == 'NGCproxy'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['blacklist_ok'] == True) & (list(map(InNGCproxyFootprint,tbdata['ra'])))) 
    if(sample1.DR == 'DR6'):    
        inds = np.where((tbdata['filter'] == sample1.band)) 
    elif(sample.DR == 'DR5'):
        if(sample.survey == 'DECaLS'):
            inds = np.where((tbdata['filter'] == band)) 
        elif(sample.survey == 'DEShyb'):
            inds = np.where((tbdata['filter'] == band) & (list(map(InDEShybFootprint,tbdata['ra'],tbdata['dec']))))
        elif(sample.survey == 'NGCproxy'):
            inds = np.where((tbdata['filter'] == band) & (list(map(InNGCproxyFootprint,tbdata['ra'])))) 

  
    #number of ccds at each point 
    nccd1=np.ones(len(tbdata))
    catalogue_name=sample1.catalog
    band = sample1.band
    sample_names = ['band_'+band]
    localdir=sample1.localdir

    # What properties do you want mapped?
    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    quicksipVerbose(sample1.verbose)
    propertiesandoperations = [('nccd1','','total')]
 
    # What properties to keep when reading the images? 
    #Should at least contain propertiesandoperations and the image corners.
    # MARCM - actually no need for ra dec image corners.   
    # Only needs ra0 ra1 ra2 ra3 dec0 dec1 dec2 dec3 only if fast track appropriate quicksip subroutines were implemented 
    #propertiesToKeep = [ 'filter', 'AIRMASS', 'FWHM','mjd_obs'] \
    #	+ ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height']
    propertiesToKeep = ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height']
       
    # Create big table with all relevant properties. 
    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [nccd1], names = propertiesToKeep + [ 'nccd1'])
    
    # Read the table, create Healtree, project it into healpix maps, and write these maps.
    # Done with Quicksip library, note it has quite a few hardcoded values (use new version by MARCM for BASS and MzLS) 
    # project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside)
    project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, localdir, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
 
    # ----- sample2 ------ 
    tbdata = pyfits.open(sample2.ccds)[1].data
    if(sample2.DR == 'DR3'):
        inds = np.where((tbdata['filter'] == sample2.band) & (tbdata['blacklist_ok'] == True)) 
    if(sample2.DR == 'DR4'):    
        inds = np.where((tbdata['filter'] == sample2.band) & (tbdata['bitmask'] == 0)) 
    if(sample2.DR == 'DR5'):
        inds = np.where((tbdata['filter'] == sample2.band) & (tbdata['blacklist_ok'] == True)) 
    if(sample2.DR == 'DR6'):    
        inds = np.where((tbdata['filter'] == sample2.band)) 
    if(sample2.DR == 'DR7'):
        inds = np.where((tbdata['filter'] == sample2.band)) 

    #number of ccds at each point 
    nccd2=np.ones(len(tbdata))
    catalogue_name=sample2.catalog
    band = sample2.band
    sample_names = ['band_'+band]
    localdir=sample2.localdir

    # Quick sip projections 
    quicksipVerbose(sample2.verbose)
    propertiesandoperations = [('nccd2','','total')]
    propertiesToKeep = ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height']
    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [nccd2], names = propertiesToKeep + [ 'nccd2'])
    project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, localdir, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
 
    # --- read ccd counts per pixel sample 1 ----
    prop='nccd1'
    op='total'
    localdir=sample1.localdir
    band=sample1.band
    catalogue_name1=sample1.catalog
    
    fname=localdir+catalogue_name1+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+catalogue_name1+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
    f = fitsio.read(fname)

    val1 = f['SIGNAL']
    pix1 = f['PIXEL']
    npix1 = np.size(pix1)
    
    # ---- read ccd counts per pixel sample 2 ----
    prop='nccd2'
    opt='total'
    localdir=sample2.localdir
    band=sample2.band
    catalogue_name2=sample2.catalog
    
    fname=localdir+catalogue_name2+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+catalogue_name2+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
    f = fitsio.read(fname)
    
    val2 = f['SIGNAL']
    pix2 = f['PIXEL']
    npix2 = np.size(pix2)
    
    # ---- compute common healpix pixels               #this is python 2.7 use isin for 3.X
    commonpix = np.in1d(pix1,pix2,assume_unique=True)  #unique: no pixel indicies are repeated
    npix=np.sum(commonpix)                             #sum of bolean in array size pix1 
    
    area1 = npix1/(float(nside)**2.*12)*360*360./pi
    area2 = npix2/(float(nside)**2.*12)*360*360./pi
    area  = npix/(float(nside)**2.*12)*360*360./pi
    print('Area of sample', sample1.catalog,' is ', np.round(area1) , ' sq. deg.')
    print('Area of sample', sample2.catalog,' is ', np.round(area2) , ' sq. deg.')
    print('The total JOINT area is ', np.round(area1+area2-area) ,'sq. deg.')
    print('The INTERSECTING area is ', np.round(area) , ' sq. deg.')
   


    # ----- plot join area using infomration from sample1 --- 
    ral = []
    decl = []
    myval = []
    for i in range(0,len(pix1)):
        if(commonpix[i]): 
           th,phi = hp.pix2ang(int(nside),pix1[i])
           ra,dec = thphi2radec(th,phi)
           ral.append(ra)
           decl.append(dec) 
           myval.append(1) 
      
    mylabel = 'join-area' 
    import matplotlib.cm as cm
    mapa = plt.scatter(ral,decl,c=myval, cmap=cm.rainbow,s=2.,lw=0,edgecolors='none')
    
    fname = localdir+mylabel+'_'+band+'_'+catalogue_name1+'_'+catalogue_name2+str(nside)+'.png'
    cbar = plt.colorbar(mapa)
    plt.xlabel('r.a. (degrees)')
    plt.ylabel('declination (degrees)')
    plt.title('Map of '+ mylabel +' for '+catalogue_name1+' and '+catalogue_name2+' '+band+'-band; Area = '+str(area))
    plt.xlim(0,360)
    plt.ylim(-30,90)
    plt.savefig(fname)
    plt.close()

    return fname

def val3p4c_seeing(sample,passmin=3,nbin=100,nside=1024):    
    """
       Requirement V4.1: z-band image quality will be smaller than 1.3 arcsec FWHM in at least one pass.
       
       Produces FWHM maps and histograms for visual inspection
    
       H-JS's addition to MARCM stable version.
       MARCM stable version, improved from AJR quick hack 
       This now included extinction from the exposures
       Uses quicksip subroutines from Boris, corrected 
       for a bug I found for BASS and MzLS ccd orientation
    """
    #nside = 1024       # Resolution of output maps
    nsideSTR=str(nside)
    #nsideSTR='1024'    # same as nside but in string format
    nsidesout = None   # if you want full sky degraded maps to be written
    ratiores = 1       # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
    mode = 1           # 1: fully sequential, 2: parallel then sequential, 3: fully parallel
    pixoffset = 0      # How many pixels are being removed on the edge of each CCD? 15 for DES.
    oversamp='1'       # ratiores in string format
    
    band = sample.band
    catalogue_name = sample.catalog
    fname = sample.ccds    
    localdir = sample.localdir
    extc = sample.extc

    #Read ccd file 
    tbdata = pyfits.open(fname)[1].data
    
    # ------------------------------------------------------
    # Obtain indices
    auxstr='band_'+band
    sample_names = [auxstr]
    if(sample.DR == 'DR3'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
    elif(sample.DR == 'DR4'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['bitmask'] == 0))
    elif(sample.DR == 'DR5'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
    elif(sample.DR == 'DR6'):
        inds = np.where((tbdata['filter'] == band))
    elif(sample.DR == 'DR7'):
        inds = np.where((tbdata['filter'] == band))

    #Read data 
    #obtain invnoisesq here, including extinction 
    nmag = Magtonanomaggies(tbdata['galdepth']-extc*tbdata['EBV'])/5.
    ivar= 1./nmag**2.

    # What properties do you want mapped?
    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    propertiesandoperations = [
    ('FWHM', '', 'min'),
    ('FWHM', '', 'num')]

    # What properties to keep when reading the images? 
    # Should at least contain propertiesandoperations and the image corners.
    # MARCM - actually no need for ra dec image corners.   
    # Only needs ra0 ra1 ra2 ra3 dec0 dec1 dec2 dec3 only if fast track appropriate quicksip subroutines were implemented 
    propertiesToKeep = [ 'filter', 'AIRMASS', 'FWHM','mjd_obs'] \
        + ['RA', 'DEC', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2','width','height']

    # Create big table with all relevant properties. 

    #tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar], names = propertiesToKeep + [ 'ivar'])
    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep], names = propertiesToKeep)

    # Read the table, create Healtree, project it into healpix maps, and write these maps.
    # Done with Quicksip library, note it has quite a few hardcoded values (use new version by MARCM for BASS and MzLS) 
    # project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside)
    project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, localdir, sample_names, inds, nside, ratiores, pixoffset, nsidesout)

      # Read Haelpix maps from quicksip  
    prop='FWHM'
    op='min'
    vmin=21.0
    vmax=24.0

    fname2=localdir+catalogue_name+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+\
           catalogue_name+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
    f = fitsio.read(fname2)

    prop='FWHM'
    op1='num'

    fname2=localdir+catalogue_name+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+\
           catalogue_name+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op1+'.fits.gz'
    f1 = fitsio.read(fname2)
    # HEALPIX DEPTH MAPS 
    # convert ivar to depth 
    import healpy as hp
    from healpix3 import pix2ang_ring,thphi2radec

    ral = []
    decl = []
    valf = []
    val = f['SIGNAL']
    npass = f1['SIGNAL']
    pixelarea = (180./np.pi)**2*4*np.pi/(12*nside**2)
    pix = f['PIXEL']
    print(min(val),max(val))
    print(min(npass),max(npass))
    # Obtain values to plot 

    j = 0
    for i in range(0,len(f)):
        if (npass[i] >= passmin):
        #th,phi = pix2ang_ring(4096,f[i]['PIXEL'])
            th,phi = hp.pix2ang(nside,f[i]['PIXEL'])
            ra,dec = thphi2radec(th,phi)
            ral.append(ra)
            decl.append(dec)
            valf.append(val[i])

    print(len(val),len(valf))
#   print len(val),len(valf),valf[0],valf[1],valf[len(valf)-1]
    minv = np.min(valf)
    maxv = np.max(valf)
    print(minv,maxv)

# Draw the map    
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm

    mylabel= prop + op

    mapa = plt.scatter(ral,decl,c=valf, cmap=cm.rainbow,s=2., vmin=minv, vmax=maxv, lw=0,edgecolors='none')
    cbar = plt.colorbar(mapa)
    plt.xlabel('r.a. (degrees)')
    plt.ylabel('declination (degrees)')
    plt.title('Map of '+ mylabel +' for '+catalogue_name+' '+band+'-band pass >='+str(passmin))
    plt.xlim(0,360)
    plt.ylim(-30,90)
    mapfile=localdir+mylabel+'pass'+str(passmin)+'_'+band+'_'+catalogue_name+str(nside)+'map.png'
    print('saving plot to ', mapfile)
    plt.savefig(mapfile)
    plt.close()
    #plt.show()
    #cbar.set_label(r'5$\sigma$ galaxy depth', rotation=270,labelpad=1)
    #plt.xscale('log')


    npix=len(pix)

    print('The total area is ', npix/(float(nside)**2.*12)*360*360./pi, ' sq. deg.')

# Draw the histogram
    minN = minv-0.0001
    maxN = maxv+.0001
    hl = np.zeros((nbin))
    for i in range(0,len(valf)):
        bin = int(nbin*(valf[i]-minN)/(maxN-minN))
        hl[bin] += 1
    Nl = []
    for i in range(0,len(hl)):
                Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))
#When an array object is printed or converted to a string, it is represented as array(typecode, initializer)
    NTl = np.array(valf)
    Nl = np.array(Nl)*0.262 # in arcsec
    hl = np.array(hl)*pixelarea
    hlcs = np.sum(hl)-np.cumsum(hl)
    print("A total of ",np.sum(hl),"squaredegrees with pass >=",str(passmin))
#    print "#FWHM  Area(>FWHM)"
#    print np.column_stack((Nl,hlcs))
    idx = (np.abs(Nl-1.3)).argmin()
    print("#FWHM  Area(>FWHM) Fractional Area(>FWHM)")
    print(Nl[idx], hlcs[idx], hlcs[idx]/hlcs[0])
    mean = np.sum(NTl)/float(len(NTl))
    std = sqrt(np.sum(NTl**2.)/float(len(NTl))-mean**2.)
    print("#mean  STD")
    print(mean,std)
    plt.plot(Nl,hl,'k-')
    #plt.xscale('log')
    plt.xlabel(op+' '+band+ ' seeing (")')
    plt.ylabel('Area (squaredegrees)')
    plt.title('Historam of '+mylabel+' for '+catalogue_name+' '+band+'-band: pass >='+str(passmin))
    ax2 = plt.twinx()
    ax2.plot(Nl,hlcs,'r')
    y0 = np.arange(0,10000, 100)
    x0 = y0*0+1.3
    shlcs = '%.2f' % (hlcs[idx])
    shlcsr = '%.2f' % (hlcs[idx]/hlcs[0])
    print("Area with FWHM greater than 1.3 arcsec fraction")
    print(shlcs, shlcsr)
    ax2.plot(x0,y0,'r--', label = r'$Area_{\rm FWHM > 1.3arcsec}= $'+shlcs+r'$deg^2$ ( $f_{\rm fail}=$'+ shlcsr+')')
    legend = ax2.legend(loc='upper center', shadow=True)
    ax2.set_ylabel('Cumulative Area sqdegrees', color='r')
    #plt.show()

    histofile = localdir+mylabel+'pass'+str(passmin)+'_'+band+'_'+catalogue_name+str(nside)+'histo.png'
    print('saving plot to ', histofile)
    plt.savefig(histofile)
#    fig.savefig(histofile)
    plt.close()

    return mapfile,histofile


def val3p4c_seeingplots(sample,passmin=3,nbin=100,nside=1024):
    """
       Requirement V4.1: z-band image quality will be smaller than 1.3 arcsec FWHM in at least one pass.
       
       Produces FWHM maps and histograms for visual inspection
    
       H-JS's addition to MARCM stable version.
       MARCM stable version, improved from AJR quick hack 
       This now included extinction from the exposures
       Uses quicksip subroutines from Boris, corrected 
       for a bug I found for BASS and MzLS ccd orientation
       
       The same as val3p4c_seeing, but this makes only plots without regenerating input files for the plots
    """
    #nside = 1024       # Resolution of output maps
    nsideSTR=str(nside)
    #nsideSTR='1024'    # same as nside but in string format
    nsidesout = None   # if you want full sky degraded maps to be written
    ratiores = 1       # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
    mode = 1           # 1: fully sequential, 2: parallel then sequential, 3: fully parallel
    pixoffset = 0      # How many pixels are being removed on the edge of each CCD? 15 for DES.
    oversamp='1'       # ratiores in string format
   
    band = sample.band
    catalogue_name = sample.catalog
    fname = sample.ccds
    localdir = sample.localdir
    extc = sample.extc

    #Read ccd file 
    tbdata = pyfits.open(fname)[1].data

  # ------------------------------------------------------
    # Obtain indices
    auxstr='band_'+band
    sample_names = [auxstr]
    if(sample.DR == 'DR3'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True))
    elif(sample.DR == 'DR4'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['bitmask'] == 0))
    elif(sample.DR == 'DR5'):
        if(sample.survey == 'DECaLS'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
        elif(sample.survey == 'DEShyb'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True) & (list(map(InDEShybFootprint,tbdata['ra'],tbdata['dec']))))
        elif(sample.survey == 'NGCproxy'):
            inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True) & (list(map(InNGCproxyFootprint,tbdata['ra'])))) 
    elif(sample.DR == 'DR6'):
        inds = np.where((tbdata['filter'] == band))
    elif(sample.DR == 'DR7'):
        if(sample.survey == 'DECaLS'):
            inds = np.where((tbdata['filter'] == band)) 
        elif(sample.survey == 'DEShyb'):
            inds = np.where((tbdata['filter'] == band) & (list(map(InDEShybFootprint,tbdata['ra'],tbdata['dec']))))
        elif(sample.survey == 'NGCproxy'):
            inds = np.where((tbdata['filter'] == band) & (list(map(InNGCproxyFootprint,tbdata['ra'])))) 


    #Read data 
    #obtain invnoisesq here, including extinction 
    nmag = Magtonanomaggies(tbdata['galdepth']-extc*tbdata['EBV'])/5.
    ivar= 1./nmag**2.

    # What properties do you want mapped?
    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    propertiesandoperations = [
    ('FWHM', '', 'min'),
    ('FWHM', '', 'num')]

    # Read Haelpix maps from quicksip  
    print("Generating only plots using the existing intermediate outputs")
    prop='FWHM'
    op='min'
    vmin=21.0
    vmax=24.0

    fname2=localdir+catalogue_name+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+\
           catalogue_name+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz'
    f = fitsio.read(fname2)

    prop='FWHM'
    op1='num'

    fname2=localdir+catalogue_name+'/nside'+nsideSTR+'_oversamp'+oversamp+'/'+\
           catalogue_name+'_band_'+band+'_nside'+nsideSTR+'_oversamp'+oversamp+'_'+prop+'__'+op1+'.fits.gz'
    f1 = fitsio.read(fname2)
    # HEALPIX DEPTH MAPS 
    # convert ivar to depth 
    import healpy as hp
    from healpix3 import pix2ang_ring,thphi2radec

    ral = []
    decl = []
    valf = []
    val = f['SIGNAL']
    npass = f1['SIGNAL']
    pixelarea = (180./np.pi)**2*4*np.pi/(12*nside**2)
    pix = f['PIXEL']
    print(min(val),max(val))
    print(min(npass),max(npass))
    # Obtain values to plot 

    j = 0
    for i in range(0,len(f)):
        if (npass[i] >= passmin):
        #th,phi = pix2ang_ring(4096,f[i]['PIXEL'])
            th,phi = hp.pix2ang(nside,f[i]['PIXEL'])
            ra,dec = thphi2radec(th,phi)
            ral.append(ra)
            decl.append(dec)
            valf.append(val[i])

    print(len(val),len(valf))
#   print len(val),len(valf),valf[0],valf[1],valf[len(valf)-1]
    minv = np.min(valf)
    maxv = np.max(valf)
    print(minv,maxv)

# Draw the map    
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm

    mylabel= prop + op

    mapa = plt.scatter(ral,decl,c=valf, cmap=cm.rainbow,s=2., vmin=minv, vmax=maxv, lw=0,edgecolors='none')
    cbar = plt.colorbar(mapa)
    plt.xlabel('r.a. (degrees)')
    plt.ylabel('declination (degrees)')
    plt.title('Map of '+ mylabel +' for '+catalogue_name+' '+band+'-band pass >='+str(passmin))
    plt.xlim(0,360)
    plt.ylim(-30,90)
    mapfile=localdir+mylabel+'pass'+str(passmin)+'_'+band+'_'+catalogue_name+str(nside)+'map.png'
    print('saving plot to ', mapfile)
    plt.savefig(mapfile)
    plt.close()
    #plt.show()
    #cbar.set_label(r'5$\sigma$ galaxy depth', rotation=270,labelpad=1)
    #plt.xscale('log')

    npix=len(pix)
    print('The total area is ', npix/(float(nside)**2.*12)*360*360./pi, ' sq. deg.')

# Draw the histogram
    minN = minv-0.0001
    maxN = maxv+.0001
    hl = np.zeros((nbin))
    for i in range(0,len(valf)):
        bin = int(nbin*(valf[i]-minN)/(maxN-minN))
        hl[bin] += 1
    Nl = []
    for i in range(0,len(hl)):
                Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))

#When an array object is printed or converted to a string, it is represented as array(typecode, initializer)
    NTl = np.array(valf)
    Nl = np.array(Nl)*0.262 # in arcsec
    hl = np.array(hl)*pixelarea
    hlcs = np.sum(hl)-np.cumsum(hl)
    print("A total of ",np.sum(hl),"squaredegrees with pass >=",str(passmin))
#    print "#FWHM  Area(>FWHM)"
#    print np.column_stack((Nl,hlcs))
    idx = (np.abs(Nl-1.3)).argmin()
    print("#FWHM  Area(>FWHM) Fractional Area(>FWHM)")
    print(Nl[idx], hlcs[idx], hlcs[idx]/hlcs[0])
    mean = np.sum(NTl)/float(len(NTl))
    std = sqrt(np.sum(NTl**2.)/float(len(NTl))-mean**2.)
    print("#mean  STD")
    print(mean,std)
    plt.plot(Nl,hl,'k-')
    #plt.xscale('log')
    plt.xlabel(op+' '+band+ ' seeing (")')
    plt.ylabel('Area (squaredegrees)')
    plt.title('Historam of '+mylabel+' for '+catalogue_name+' '+band+'-band: pass >='+str(passmin))
    ax2 = plt.twinx()
    ax2.plot(Nl,hlcs,'r')
    y0 = np.arange(0,10000, 100)
    x0 = y0*0+1.3
    shlcs = '%.2f' % (hlcs[idx])
    shlcsr = '%.2f' % (hlcs[idx]/hlcs[0])
    print("Area with FWHM greater than 1.3 arcsec  fraction")
    print(shlcs, shlcsr)
    ax2.plot(x0,y0,'r--', label = r'$Area_{\rm FWHM > 1.3arcsec}= $'+shlcs+r'$deg^2$ ( $f_{\rm fail}=$'+ shlcsr+')')
    legend = ax2.legend(loc='upper center', shadow=True)
    ax2.set_ylabel('Cumulative Area sqdegrees', color='r')
    #plt.show()
#    fig = plt.gcf()
    histofile = localdir+mylabel+'pass'+str(passmin)+'_'+band+'_'+catalogue_name+str(nside)+'histo.png'
    print('saving plot to ', histofile)
    plt.savefig(histofile)
#    fig.savefig(histofile)
    plt.close()

    return mapfile,histofile



