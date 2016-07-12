"""
unit test script for functions in decals_sim.py
-- creates a tim object and sims stamp 
-- allows user to play with these on command line and confirm masking, invver behaves as expected, fluxes are correct, etc.

RUN:
python legacyanalysis/decals_sim_test_wone_tim.py
or
ipython
%run legacyanalysis/decals_sim_test_wone_tim.py

USE:
run with ipython then can play with tim, stamp objects on command line!!
"""

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')
import os
import sys
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.table import Table, Column, vstack
import galsim
import photutils

###
from tractor.psfex import PsfEx, PixelizedPsfEx
from tractor import Tractor
from tractor.basics import (NanoMaggies, PointSource, GaussianMixtureEllipsePSF,PixelizedPSF, RaDecPos)
from astrometry.util.fits import fits_table
from legacypipe.common import LegacySurveyData,wcs_for_brick
###
from legacyanalysis.decals_sim import SimImage,SimDecals,BuildStamp,build_simcat

def get_metacat_simcat_tim(brickname, W=200,H=200,pixscale=0.262,verbose=0, splinesky=False):
    '''return metacat,simcat, one tim object
    metacat,simcat -- from decals_sim
    tim object -- for same ra,dec range of metacat,simcat'''
    
    survey = LegacySurveyData()
    survey = SimDecals(metacat=metacat, simcat=simcat, output_dir=output_dir)
    brick = survey.get_brick_by_name(brickname)
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
    if ccds is None:
        raise NothingToDoError('No CCDs touching brick')
    print(len(ccds), 'CCDs touching target WCS')
    # Sort images by band -- this also eliminates images whose
    # *image.filter* string is not in *bands*.
    print('Unique filters:', np.unique(ccds.filter))
    bands='grz'
    ccds.cut(np.hstack([np.flatnonzero(ccds.filter == band) for band in bands]))
    print('Cut on filter:', len(ccds), 'CCDs remain.')
    
    print('Cutting out non-photometric CCDs...')
    I = survey.photometric_ccds(ccds)
    print(len(I), 'of', len(ccds), 'CCDs are photometric')
    ccds.cut(I)
    #just first ccd
    ccd= ccds[0]
    #get tim object
    im = survey.get_image_object(ccd)
    get_tim_kwargs = dict(pixPsf=True, splinesky=splinesky)
    tim = im.get_tractor_image(**get_tim_kwargs)
    return tim,targetwcs,ccd

def get_metacat(brickname,objtype,nobj,chunksize,nchunk,zoom,rmag_range):
    '''following decals_sim'''
    metacols = [
        ('BRICKNAME', 'S10'),
        ('OBJTYPE', 'S10'),
        ('NOBJ', 'i4'),
        ('CHUNKSIZE', 'i2'),
        ('NCHUNK', 'i2'),
        ('ZOOM', 'i4', (4,)),
        ('SEED', 'S20'),
        ('RMAG_RANGE', 'f4', (2,))]
    metacat = Table(np.zeros(1, dtype=metacols))

    metacat['BRICKNAME'] = brickname
    metacat['OBJTYPE'] = objtype
    metacat['NOBJ'] = nobj
    metacat['CHUNKSIZE'] = chunksize
    metacat['NCHUNK'] = nchunk
    metacat['ZOOM'] = zoom
    metacat['RMAG_RANGE'] = rmag_range    
    return metacat

def plot_tim(tim):
    '''basic plotting func'''
    fig = plt.figure(figsize=(5,10))
    ax = fig.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.imshow(tim.getImage(), **tim.ima)
    ax.axis('off')
    fig.savefig('./test.png',bbox_inches='tight')

def check_poisson_noise(stamp,ivarstamp,objstamp):
    '''each pixel of stamp+noise image - stamp image should be gaussian distributed with std dev = sqrt(pix value in stamp)'''
    diff=np.zeros((stamp.array.shape[0],stamp.array.shape[1],1000))
    for cnt in range(diff.shape[-1]):
        stamp_copy= stamp.copy()
        ivarstamp_copy= ivarstamp.copy()
        stamp_copy, ivarstamp_copy = objstamp.addnoise(stamp_copy, ivarstamp_copy)
        diff[:,:,cnt]= stamp_copy.array-stamp.array
    one_std= np.sqrt( np.sqrt(stamp.array**2)) 
    for x in np.arange(stamp.array.shape[0])[::4]:
        for y in np.arange(stamp.array.shape[1])[::4]:
            junk= plt.hist(diff[x,y,:],range=(-2*one_std[x,y],2*one_std[x,y]))
            plt.savefig('x%d_y%d_hist.png')
            plt.close()

#def main():
parser = ArgumentParser()
parser.add_argument('-b', '--brickname', default='2428p117', help='exposure number')
parser.add_argument('--splinesky', action='store_true', help='Use spline sky model?')
parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0,
                    help='Toggle on verbose output')
args = parser.parse_args()

#tim object
tim,brickwcs,ccd= get_one_tim(args.brickname,W=3600,H=3600,pixscale=0.262,verbose=args.verbose, splinesky=True)
#setup sims
nobj,seed = 500,None
metacat= get_metacat(args.brickname,'STAR',nobj,500,1,(0,3600,0,3600),(18, 26))
simcat = build_simcat(nobj, args.brickname, brickwcs, metacat, seed)
stamp_builder = BuildStamp(tim, gain=ccd.arawgain, seed=seed)
ap_flux=np.zeros(len(simcat))-1
for i,obj in enumerate(simcat):  
    stamp = stamp_builder.star(obj)

    ap_size=7. #arcsec
    pixsc=0.262 #decam
    aper=photutils.CircularAperture((obj['X'],obj['Y']),ap_size/pixsc)
    p = photutils.aperture_photometry(stamp.array, aper) # error=np.zeros(stamp.array.shape)
    ap_flux[i]= p['aperture_sum']

#stamp
#unit test after this or run from ipython to play with on command line

####
# check_poisson_noise()

####
#in decals_sim.py have a test option that if turned on makes 3panel yellow box plots, using code like below
#tim.sims_image= sims_image.array
#tim.sims_inverr= np.sqrt(sims_ivar.array)
#tim.sims_xy= tim.sims_xy.astype(int)
#tim.data = image.array + sims_image.array
#tim.inverr = np.sqrt(invvar.array + sims_ivar.array)
#plot image,image regions where have sims, just sims as 3 plot panel with yellow boxes
#basename= plots.get_basename(self.imgfn)
#plots.image_v_stamp([tim.data,tim.data-tim.sims_image,tim.sims_image], \
#                    xy_lim= tim.sims_xy, name=os.path.join(self.survey.output_dir,"image_v_stamp_%s.png" % basename))
#plots.image_v_stamp([np.power(tim.inverr,-1),np.power(tim.sims_inverr,-1)], \
#                    xy_lim= tim.sims_xy, titles=['image_std','sims_std'],\
					#name=os.path.join(self.survey.output_dir,"std_%s.png" % basename))
#print('exiting early')
#sys.exit()


#if __name__ == "__main__":
#    main()
