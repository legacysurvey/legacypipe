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
###
from tractor.psfex import PsfEx, PixelizedPsfEx
from tractor import Tractor
from tractor.basics import (NanoMaggies, PointSource, GaussianMixtureEllipsePSF,PixelizedPSF, RaDecPos)
from astrometry.util.fits import fits_table
from legacypipe.common import LegacySurveyData,wcs_for_brick
###
from legacyanalysis.decals_sim import BuildStamp,build_simcat

def get_one_tim(brickname,W=200,H=200,pixscale=0.25,verbose=0, splinesky=False):
    '''given brickname returns tim and targetwcs,ccd for that tim object'''
    survey = LegacySurveyData()
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

def get_metacat(brickname):
    '''following decals_sim'''
    metacat = Table()
    metacat['brickname'] = Column([brickname],dtype='S10')
    metacat['objtype'] = Column(['STAR'],dtype='S10')
    metacat['nobj'] = Column([1],dtype='i4')
    metacat['chunksize'] = Column([500],dtype='i2')
    metacat['nchunk'] = Column([1],dtype='i2')
    #metacat['RA'] = Column([ra_range],dtype='f8')
    #metacat['DEC'] = Column([dec_range],dtype='f8')
    metacat['rmag_range'] = Column([[18,26]],dtype='f4')
    metacat['seed'] = Column([111],dtype='i4')
    return metacat

def plot_tim(tim):
    '''basic plotting func'''
    fig = plt.figure(figsize=(5,10))
    ax = fig.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.imshow(tim.getImage(), **tim.ima)
    ax.axis('off')
    fig.savefig('./test.png',bbox_inches='tight')

#def main():
"""
Main routine.
"""

parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--brickname', default='2428p117', help='exposure number')
parser.add_argument('--splinesky', action='store_true', help='Use spline sky model?')
parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0,
                    help='Toggle on verbose output')
args = parser.parse_args()

#tim object
tim,brickwcs,ccd= get_one_tim(args.brickname,W=200,H=200,pixscale=0.25,verbose=args.verbose, splinesky=True)
#setup sims
metacat= get_metacat(args.brickname)
nobj,seed = 1,111
simcat = build_simcat(nobj, args.brickname, brickwcs, metacat, seed)
#stamp
objstamp = BuildStamp(tim, gain=ccd.arawgain, seed=seed)
stamp = objstamp.star(simcat[0])
#unit test after this or run from ipython to play with on command line


#if __name__ == "__main__":
#    main()
