#!/usr/bin/env python

"""Insert artificial (fake) galaxies into the DECaLS imaging and reprocess
through The Tractor.

TODO:
* Model a mixture of object types.
"""
from __future__ import division, print_function

import os
import sys
import logging
import argparse
import numpy as np

import galsim
from astropy.io import fits

from projects.desi.common import *

# Global variables.
decals_dir = os.getenv('DECALS_DIR')
fake_decals_dir = os.getenv('FAKE_DECALS_DIR')

logging.basicConfig(format='%(message)s',level=logging.INFO,stream=sys.stdout)
log = logging.getLogger('decals_simulations')

def get_brickinfo(brickname=None):
    """Get info on this brick.

    """
    allbrickinfo = fits.getdata(os.path.join(decals_dir,'decals-bricks.fits'),1)
    brickinfo = allbrickinfo[np.where((allbrickinfo['brickname']==brickname)*1==1)]

    decals = Decals()
    wcs = wcs_for_brick(decals.get_brick_by_name(brickname))

    return brickinfo, wcs

def get_ccdinfo(brickwcs=None):
    """Get info on this brick and on the CCDs touching it.

    """
    allccdinfo =  fits.getdata(os.path.join(decals_dir,'decals-ccds-zeropoints.fits'))

    # Get all the CCDs that touch this brick
    decals = Decals()
    these = ccds_touching_wcs(brickwcs,decals.get_ccds())
    #ccdinfo = decals.ccds_touching_wcs(targetwcs)
    ccdinfo = allccdinfo[these]

    log.info('Got {} CCDs'.format(len(ccdinfo)))
    return ccdinfo

def build_priors(nobj=20,brickname=None,objtype='ELG',ra_range=None,
                 dec_range=None,seed=None):
    """Choose priors according to the type of object.  Will eventually generalize
       this so that a mixture of object types can be simulated.

    """

    from astropy.table import Table, Column
    rand = np.random.RandomState(seed=seed)

    # Assign central coordinates uniformly
    ra = rand.uniform(ra_range[0],ra_range[1],nobj)
    dec = rand.uniform(dec_range[0],dec_range[1],nobj)

    if objtype.upper()=='ELG':
        # Disk parameters
        disk_n_range = [1.0,1.0]
        disk_r50_range = [0.5,2.5]
        disk_ba_range = [0.2,1.0]

        disk_n = rand.uniform(disk_n_range[0],disk_n_range[1],nobj)
        disk_r50 = rand.uniform(disk_r50_range[0],disk_r50_range[1],nobj)
        disk_ba = rand.uniform(disk_ba_range[0],disk_ba_range[1],nobj)
        disk_phi = rand.uniform(0,180,nobj)

        ## Bulge parameters
        #bulge_r50_range = [0.1,1.0]
        #bulge_n_range = [3.0,5.0]
        #bdratio_range = [0.0,1.0] # bulge-to-disk ratio

        # Magnitudes and colors
        rmag_range = [18.0,24.0]
        gr_range = [-0.3,0.5]
        rz_range = [0.0,1.5]

        rmag = rand.uniform(rmag_range[0],rmag_range[1],nobj)
        gr = rand.uniform(gr_range[0],gr_range[1],nobj)
        rz = rand.uniform(rz_range[0],rz_range[1],nobj)

    # For convenience, also store the grz fluxes in nanomaggies.
    gflux = 1E9*10**(-0.4*(rmag+gr))
    rflux = 1E9*10**(-0.4*rmag)
    zflux = 1E9*10**(-0.4*(rmag-rz))

    # Pack into a Table.
    priors = Table()
    priors['ID'] = Column(np.arange(nobj,dtype='i4'))
    priors['RA'] = Column(ra,dtype='f8')
    priors['DEC'] = Column(dec,dtype='f8')
    priors['R'] = Column(rmag,dtype='f4')
    priors['GR'] = Column(gr,dtype='f4')
    priors['RZ'] = Column(rz,dtype='f4')
    priors['GFLUX'] = Column(gflux,dtype='f4')
    priors['RFLUX'] = Column(rflux,dtype='f4')
    priors['ZFLUX'] = Column(zflux,dtype='f4')
    priors['DISK_N'] = Column(disk_n,dtype='f4')
    priors['DISK_R50'] = Column(disk_r50,dtype='f4')
    priors['DISK_BA'] = Column(disk_ba,dtype='f4')
    priors['DISK_PHI'] = Column(disk_phi,dtype='f4')

    # Write out.
    outfile = os.path.join(fake_decals_dir,'priors_'+brickname+'.fits')
    log.info('Writing {}'.format(outfile))
    if os.path.isfile(outfile):
        os.remove(outfile)
    priors.write(outfile)

    return priors

def copyfiles(ccdinfo=None):
    """Copy the CP-processed images, inverse variance maps, and bad-pixel masks we
    need from DECALS_DIR to FAKE_DECALS_DIR, creating directories as necessary.

    """
    from distutils.file_util import copy_file

    allcpimage = ccdinfo['CPIMAGE']
    allcpdir = set([cpim.split('/')[1] for cpim in allcpimage])
    
    log.info('Creating directories...')
    for cpdir in list(allcpdir):
        outdir = os.path.join(fake_decals_dir,'images','decam',cpdir)
        if not os.path.isdir(outdir):
            log.info('   {}'.format(outdir))
            os.makedirs(outdir)
    
    log.info('Copying files...')
    for cpimage in list(set(allcpimage)):
        cpdir = cpimage.split('/')[1]
        indir = os.path.join(decals_dir,'images','decam',cpdir)
        outdir = os.path.join(fake_decals_dir,'images','decam',cpdir)

        imfile = cpimage.split('/')[2].split()[0]
        log.info('  {}'.format(imfile))
        copy_file(os.path.join(indir,imfile),os.path.join(outdir,imfile),update=0)

        imfile = imfile.replace('ooi','oow')
        #log.info('{}-->{}'.format(os.path.join(indir,imfile),os.path.join(outdir,imfile)))
        copy_file(os.path.join(indir,imfile),os.path.join(outdir,imfile),update=0)

        imfile = imfile.replace('oow','ood')
        #log.info('{}-->{}'.format(os.path.join(indir,imfile),os.path.join(outdir,imfile)))
        copy_file(os.path.join(indir,imfile),os.path.join(outdir,imfile),update=0)

class simobj_info():
    from tractor import psfex
    def __init__(self,ccdinfo):
        """Access everything we need about an individual CCD.
        
        """        
        self.cpimage = ccdinfo['CPIMAGE']
        self.cpimage_hdu = ccdinfo['CPIMAGE_HDU']
        self.calname = ccdinfo['CALNAME']
        self.filter = ccdinfo['FILTER']
        self.imfile = os.path.join(fake_decals_dir,'images',self.cpimage)
        self.ivarfile = self.imfile.replace('ooi','oow')
        self.wcsfile = os.path.join(decals_dir,'calib','decam',
                                    'astrom-pv',self.calname+'.wcs.fits')
        self.psffile = os.path.join(decals_dir,'calib','decam',
                                    'psfex',self.calname+'.fits')
        self.magzpt = float(ccdinfo['CCDZPT'] + 2.5*np.log10(ccdinfo['EXPTIME']))
        self.gain = float(ccdinfo['ARAWGAIN']) # [electron/ADU]

    def getdata(self):
        """Read the CCD image and inverse variance data, and the corresponding headers. 
        
        """
        #log.info('Reading extension {} of image {}'.format(self.ccdnum,self.imfile))
        image = galsim.fits.read(self.imfile,hdu=self.cpimage_hdu)       # [ADU]
        invvar = galsim.fits.read(self.ivarfile,hdu=self.cpimage_hdu) # [1/ADU^2]
        
        imhdr = galsim.fits.FitsHeader(self.imfile,hdu=self.cpimage_hdu)
        ivarhdr = galsim.fits.FitsHeader(self.ivarfile,hdu=self.cpimage_hdu)
        
        self.width = image.xmax
        self.height = image.ymax
        
        return image, invvar, imhdr, ivarhdr

    def getwcs(self):
        """Read the global WCS for this CCD."""
        wcs, origin = galsim.wcs.readFromFitsHeader(
            galsim.fits.FitsHeader(self.wcsfile))
        self.wcs = wcs

        return wcs
    
    def getpsf(self):
        """Read the PSF for this CCD."""
        from tractor.basics import GaussianMixtureEllipsePSF
        psf = psfex.PsfEx(self.psffile,self.width,self.height,ny=13,nx=7,
                          psfClass=GaussianMixtureEllipsePSF,K=2)
        self.psf = psf

        return psf

    def getlocalwcs(self,image_pos=None):
        """Get the local WCS, given a position."""
        localwcs = self.wcs.local(image_pos=image_pos)
        pixscale, shear, theta, flip = localwcs.getDecomposition() # get the pixel scale

        return localwcs, pixscale

    def getlocalpsf(self,image_pos=None,pixscale=0.262):
        """Get the local PSF, given a position.  Need to recentroid because this is a
        PSFeX PSF."""
        xpos = int(image_pos.x)
        ypos = int(image_pos.y)
        psfim = PsfEx.instantiateAt(self.psf,xpos,ypos)[5:-5,5:-5] # trim
        psf = galsim.InterpolatedImage(galsim.Image(psfim),scale=pixscale,flux=1.0)
        psf_centroid = psf.centroid()
        psf = psf.shift(-psf_centroid.x,-psf_centroid.y)

        return psf
    
    def getobjflux(self,objinfo):
        """Calculate the flux of a given object in ADU."""
        flux = objinfo[self.filter.upper()+'FLUX']
        flux *= 10**(0.4*(self.magzpt-22.5)) # [ADU]

        return float(flux)

    
def insert_simobj(gsparams=None,priors=None,ccdinfo=None):
    """Simulate objects and place them into individual CCDs.

    """
    stampwidth = 45 # postage stamp width [pixels, roughly 14 arcsec]
    stampbounds = galsim.BoundsI(-stampwidth,stampwidth,-stampwidth,stampwidth)
    imagebounds = galsim.BoundsI(0,2046,0,4094)
    
    band = np.array(['g','r','z'])

    # Calculate the g, r, and z band fluxes and stack them in an array.
    #flux = np.vstack([gflux,rflux,zflux])
        
    for ccd in ccdinfo:
        # Gather some basic info on this CCD and then read the data, the WCS
        # info, and initialize the PSF.
        siminfo = simobj_info(ccd)
        wcs = siminfo.getwcs()

        # Loop on each object and figure out which, if any, objects will be
        # placed on this CCD.
        onccd = []
        for iobj, objinfo in enumerate(priors):
            pos = wcs.toImage(galsim.CelestialCoord(objinfo['RA']*galsim.degrees,
                objinfo['DEC']*galsim.degrees))
            stampbounds1 = stampbounds.shift(galsim.PositionI(int(pos.x),int(pos.y)))
        
            overlap = stampbounds1 & imagebounds
            #if iobj<5:
            #    print(iobj, pos, stampbounds1, imagebounds, overlap)
            if (overlap.xmax>=0 and overlap.ymax>=0 and overlap.xmin<=imagebounds.xmax and
                overlap.ymin<=imagebounds.ymax and overlap.area()>0):
                onccd.append(iobj)

        nobj = len(onccd)
        if nobj>0:
            log.info('Adding {} objects to CCD {}'.format(nobj,siminfo.cpimage_hdu))

            image, invvar, imhdr, ivarhdr = siminfo.getdata()
            initpsf = siminfo.getpsf()
            
            for iobj in range(nobj):
                objinfo = priors[onccd[iobj]]
                pos = wcs.toImage(galsim.CelestialCoord(objinfo['RA']*galsim.degrees,
                                                        objinfo['DEC']*galsim.degrees))

                xpos = int(pos.x)
                ypos = int(pos.y)
                offset = galsim.PositionD(pos.x-xpos,pos.y-ypos)

                # Get the WCS and PSF at the center of the stamp and the integrated
                # flux of the object (in the appropriate band).
                localwcs, pixscale = siminfo.getlocalwcs(image_pos=pos)
                localpsf = siminfo.getlocalpsf(image_pos=pos,pixscale=pixscale)
                objflux = siminfo.getobjflux(objinfo)

                # Finally build the object.
                obj = galsim.Sersic(float(objinfo['DISK_N']),half_light_radius=
                                    float(objinfo['DISK_R50']),
                                    flux=objflux,gsparams=gsparams)
                obj = obj.shear(q=float(objinfo['DISK_BA']),beta=
                                float(objinfo['DISK_PHI'])*galsim.degrees)
                obj = galsim.Convolve([obj,localpsf])
            
                stamp = obj.drawImage(offset=offset,wcs=localwcs,method='no_pixel')
                stamp.setCenter(xpos,ypos)

                overlap = stamp.bounds & image.bounds
                if (overlap.xmax>=0 and overlap.ymax>=0 and overlap.xmin<=image.bounds.xmax and
                    overlap.ymin<=image.bounds.ymax and overlap.area()>0):
                    print(iobj, pos, stamp.bounds, image.bounds, overlap)
                    stamp = stamp[overlap]

                    # Add Poisson noise
                    varstamp = invvar[overlap].copy() # [1/ADU^2]
                    varstamp.invertSelf()             # [ADU^2]
                    medvar = np.median(varstamp.array[varstamp.array>0])
                    varstamp.array[varstamp.array<(0.2*medvar)] = medvar

                    # Convert to electrons
                    stamp *= siminfo.gain         # [electron]
                    varstamp *= (siminfo.gain**2) # [electron^2]

                    stamp.addNoise(galsim.VariableGaussianNoise(galsim.BaseDeviate(),varstamp))
                    varstamp += stamp
                
                    stamp /= siminfo.gain          # [ADU]
                    varstamp /= (siminfo.gain**2)  # [ADU^2]
                    varstamp.invertSelf()          # [1/ADU^2]

                    image[overlap] += stamp
                    invvar[overlap] = varstamp

            # Write out.
            log.info('Writing extension {} of image {}'.format(siminfo.cpimage_hdu,
                                                               siminfo.imfile))
            fits.update(siminfo.imfile,image.array,ext=siminfo.cpimage_hdu,
                        header=fits.Header(imhdr.items()))
            fits.update(siminfo.ivarfile,invvar.array,ext=siminfo.cpimage_hdu,
                        header=fits.Header(ivarhdr.items()))

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='DECaLS simulations.')
    parser.add_argument('-n', '--nobj', type=long, default=None, metavar='', 
                        help='number of objects to simulate (required input)')
    parser.add_argument('-b', '--brick', type=str, default='2428p117', metavar='', 
                        help='simulate objects in this brick')
    parser.add_argument('-o', '--objtype', type=str, default='ELG', metavar='', 
                        help='object type (ELG, LRG, PSF, MIX)') 
    parser.add_argument('-s', '--seed', type=long, default=None, metavar='', 
                        help='random number seed')
    parser.add_argument('--zoom', nargs=4, type=int, metavar='', 
                        help='see runbrick.py (default is to populate the full brick)')
    parser.add_argument('--no-qaplots', action='store_true',
                        help='do not generate QAplots')

    args = parser.parse_args()
    if args.nobj is None:
        parser.print_help()
        sys.exit(1)

    objtype = args.objtype.upper()
    nobj = args.nobj
    brickname = args.brick

    log.info('Working on brick {}'.format(brickname))
    log.info('Simulating {} objects of objtype={}'.format(nobj,objtype))
        
    # Get the brick info and corresponding WCS
    brickinfo, brickwcs = get_brickinfo(brickname)

    if args.zoom is None:
        ra_range = [brickinfo['ra1'],brickinfo['ra2']]
        dec_range = [brickinfo['dec1'],brickinfo['dec2']]
    else:
        pixscale = 0.262/3600.0 # average pixel scale [deg/pixel]
        zoom = args.zoom
        dx = zoom[1]-zoom[0]
        dy = zoom[3]-zoom[2]

        ra, dec = brickwcs.pixelxy2radec(zoom[0]+dx/2,zoom[2]+dy/2)
        ra_range = [ra-dx*pixscale/2,ra+dx*pixscale/2]
        dec_range = [dec-dy*pixscale/2,dec+dy*pixscale/2]

        brickwcs = brickwcs.get_subimage(zoom[0],zoom[2],dx,dy)

    # Get the CCDs in the region of interest.
    ccdinfo = get_ccdinfo(brickwcs)

    log.info('RA range: {:.6f} to {:.6f}'.format(float(ra_range[0]),float(ra_range[1])))
    log.info('DEC range: {:.6f} to {:.6f}'.format(float(dec_range[0]),float(dec_range[1])))
        
    # Build the prior parameters and make some QAplots.
    log.info('Building the PRIORS table.')
    priors = build_priors(nobj,brickname,objtype,ra_range,dec_range,
                          seed=args.seed)

    # Write out some QAplots
    if args.no_qaplots is False:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.patches import Rectangle

        color = iter(cm.rainbow(np.linspace(0,1,len(ccdinfo))))

        fig = plt.figure()
        ax = fig.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False) 
        ax.plot(priors['RA'],priors['DEC'],'gs',markersize=3)
        for ii, ccd in enumerate(ccdinfo):
            dy = ccd['WIDTH']*0.262/3600.0
            dx = ccd['HEIGHT']*0.262/3600.0
            rect = plt.Rectangle((ccd['RA']-dx/2,ccd['DEC']-dy/2),
                                 dx,dy,fill=False,lw=1,color=next(color),
                                 ls='solid')
            ax.add_patch(rect)
            rect = plt.Rectangle((brickinfo['RA1'],brickinfo['DEC1']),
                                 brickinfo['RA2']-brickinfo['RA1'],
                         brickinfo['DEC2']-brickinfo['DEC1'],fill=False,lw=3,
                                 color='b')
        ax.add_patch(rect)
        #ax.set_xlim(np.array([brickinfo['RA1'][0],brickinfo['RA2'][0]])*[0.9999,1.0001])
        ax.set_xlim(np.array([brickinfo['RA2'][0],brickinfo['RA1'][0]])*[1.0001,0.9999])
        ax.set_ylim(np.array([brickinfo['DEC1'][0],brickinfo['DEC2'][0]])*[0.99,1.01])
        ax.set_xlabel('$RA\ (deg)$',fontsize=18)
        ax.set_ylabel('$Dec\ (deg)$',fontsize=18)
        qafile = os.path.join(fake_decals_dir,'qa_'+brickname+'.pdf')

        log.info('Writing QAplot {}'.format(qafile))
        fig.savefig(qafile)

    # Copy the files we need.
    copyfiles(ccdinfo)

    # Do the simulation!
    gsparams = galsim.GSParams(maximum_fft_size=2L**30L)
    insert_simobj(gsparams=gsparams,priors=priors,ccdinfo=ccdinfo)

# python $TRACTOR_DIR/projects/desi/runbrick.py --stage image_coadds --no-write --threads 8 --gpsf --radec 242.845 11.75 --width 500 --height 500
# python decals_simulations.py -n 10 --zoom 1750 1850 1750 1850


if __name__ == "__main__":
    main()
