#!/usr/bin/env python

"""Redo the Tractor photometry of the "large" galaxies in Legacy Survey imaging.

J. Moustakas
Siena College
2016 June 6

"""
from __future__ import division, print_function

import os
import sys
import pdb
import argparse

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table, vstack

from PIL import Image, ImageDraw

from astrometry.util.util import Tan
from astrometry.util.fits import merge_tables

from legacypipe.runbrick import run_brick
from legacypipe.common import ccds_touching_wcs, LegacySurveyData

PIXSCALE = 0.262 # average pixel scale [arcsec/pix]

def _uniqccds(ccds):
    '''Get the unique set of CCD files.'''
    ccdfile = []
    [ccdfile.append('{}-{}'.format(expnum, ccdname)) for expnum,
     ccdname in zip(ccds.expnum, ccds.ccdname)]
    _, indx = np.unique(ccdfile, return_index=True)
    return indx

def _getfiles(ccds):
    '''Figure out the set of images and calibration files we need to transfer, if any.'''

    ccdinfo = ccds[_uniqccds(ccds)]
    nccd = len(ccdinfo)

    expnum = ccdinfo.expnum
    ccdname = ccdinfo.ccdname

    psffiles = []
    skyfiles = []
    imagefiles = []
    for ii in range(nccd):
        exp = '{0:08d}'.format(expnum[ii])
        rootfile = os.path.join(exp[:5], exp, 'decam-'+exp+'-'+ccdname[ii]+'.fits')
        psffiles.append(os.path.join('calib', 'decam', 'psfex', rootfile))
        skyfiles.append(os.path.join('calib', 'decam', 'splinesky', rootfile))
        imagefiles.append(os.path.join('images', str(np.core.defchararray.strip(ccdinfo.image_filename[ii]))))

    ccdfiles = open('/tmp/ccdfiles.txt', 'w')
    for ii in range(nccd):
        ccdfiles.write(psffiles[ii]+'\n')
    for ii in range(nccd):
        ccdfiles.write(skyfiles[ii]+'\n')
    for ii in range(nccd):
        ccdfiles.write(imagefiles[ii]+'\n')
    for ii in range(nccd):
        ccdfiles.write(imagefiles[ii].replace('ooi', 'oow')+'\n')
    for ii in range(nccd):
        ccdfiles.write(imagefiles[ii].replace('ooi', 'ood')+'\n')
    ccdfiles.close()

    cmd = "rsync -avP --files-from='/tmp/ccdfiles.txt' cori:/global/cscratch1/sd/desiproc/dr3/ /global/work/decam/versions/work/"
    print('You should run the following command:')
    print('  {}'.format(cmd))

def _catalog_template(nobj=1):
    from astropy.table import Table
    cols = [
        ('GALAXY', 'S20'), 
        ('RA', 'f8'), 
        ('DEC', 'f8'),
        ('RADIUS', 'f4')
        ]
    catalog = Table(np.zeros(nobj, dtype=cols))
    catalog['RADIUS'].unit = 'arcsec'

    return catalog

def _simplewcs(gal):
    '''Build a simple WCS object for a single galaxy.'''
    diam = np.ceil(gal['RADIUS']/PIXSCALE).astype('int16') # [pixels]
    galwcs = Tan(gal['RA'], gal['DEC'], diam/2+0.5, diam/2+0.5,
                 -PIXSCALE, 0.0, PIXSCALE, 0.0, 
                 float(diam), float(diam))
    return galwcs

def read_rc3():
    """Read the RC3 catalog and put it in a standard format."""
    catdir = os.getenv('CATALOGS_DIR')
    cat = fits.getdata(os.path.join(catdir, 'rc3', 'rc3_catalog.fits'), 1)

    ## For testing -- randomly pre-select a subset of galaxies.
    #nobj = 500
    #seed = 5781
    #rand = np.random.RandomState(seed)
    #these = rand.randint(0, len(cat)-1, nobj)
    #cat = cat[these]

    outcat = _catalog_template(len(cat))
    outcat['RA'] = cat['RA']
    outcat['DEC'] = cat['DEC']
    outcat['RADIUS'] = 0.1*10.0**cat['LOGD_25']*60.0/2.0 # semi-major axis diameter [arcsec]
    fix = np.where(outcat['RADIUS'] == 0.0)[0]
    if len(fix) > 0:
        outcat['RADIUS'][fix] = 30.0
    
    #plt.hist(outcat['RADIUS'], bins=50, range=(1, 300))
    #plt.show()

    for name in ('NAME1', 'NAME2', 'NAME3', 'PGC'):
        need = np.where(outcat['GALAXY'] == '')[0]
        if len(need) > 0:
            outcat['GALAXY'][need] = cat[name][need].replace(' ', '')

    # Temporarily cull the sample.
    these = np.where((outcat['RADIUS'] > 20)*(outcat['RADIUS'] < 25))[0]
    outcat = outcat[these] 

    return outcat

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--build-sample', action='store_true', help='Build the sample.')
    parser.add_argument('--viewer-cutouts', action='store_true', help='Get jpg cutouts from the viewer.')
    parser.add_argument('--ccd-cutouts', action='store_true', help='Get CCD cutouts of each galaxy.')
    parser.add_argument('--runbrick', action='store_true', help='Run the pipeline.')
    parser.add_argument('--runbrick-cutouts', action='store_true', help='Annotate the jpg cutouts from the custom pipeline.')
    parser.add_argument('--build-webpage', action='store_true', help='(Re)build the web content.')
    args = parser.parse_args()

    # Top-level directory
    key = 'LEGACY_SURVEY_LARGE_GALAXIES'
    if key not in os.environ:
        print('Required ${} environment variable not set'.format(key))
        return 0
    largedir = os.getenv(key)
    samplefile = os.path.join(largedir, 'large-galaxies-sample.fits')

    # Some convenience variables.
    objtype = ('PSF', 'SIMP', 'EXP', 'DEV', 'COMP')
    objcolor = ('white', 'red', 'orange', 'cyan', 'yellow')

    # --------------------------------------------------
    # Build the sample of large galaxies based on the available imaging.
    if args.build_sample:

        # Read the parent catalog.
        cat = read_rc3()
        
        # Create a simple WCS object for each object and find all the CCDs
        # touching that WCS footprint.
        survey = LegacySurveyData(version='dr2') # hack!
        allccds = survey.get_ccds()
        keep = np.concatenate((survey.apply_blacklist(allccds),
                               survey.photometric_ccds(allccds)))
        allccds.cut(keep)

        ccdlist = []
        outcat = []
        for gal in cat:
            galwcs = _simplewcs(gal)

            ccds1 = allccds[ccds_touching_wcs(galwcs, allccds)]
            ccds1 = ccds1[_uniqccds(ccds1)]
            
            if len(ccds1) > 0 and 'g' in ccds1.filter and 'r' in ccds1.filter and 'z' in ccds1.filter:
                print('Found {} CCDs for {}, D(25)={:.4f}'.format(
                    len(ccds1), gal['GALAXY'], gal['RADIUS']))
                
                ccdsfile = os.path.join(largedir, 'ccds', '{}-ccds.fits'.format(gal['GALAXY'].strip().lower()))
                print('  Writing {}'.format(ccdsfile))
                if os.path.isfile(ccdsfile):
                    os.remove(ccdsfile)
                ccds1.writeto(ccdsfile)
                
                ccdlist.append(ccds1)
                if len(outcat) == 0:
                    outcat = gal
                else:
                    outcat = vstack((outcat, gal))
                #if gal['GALAXY'] == 'MCG5-19-36':
                #    pdb.set_trace()

        # Write out the final catalog.
        samplefile = os.path.join(largedir, 'large-galaxies-sample.fits')
        if os.path.isfile(samplefile):
            os.remove(samplefile)
        print('Writing {}'.format(samplefile))
        outcat.write(samplefile)
        print(outcat)

        # Do we need to transfer any of the data to nyx?
        _getfiles(merge_tables(ccdlist))

    # --------------------------------------------------
    # Get data, model, and residual cutouts from the legacysurvey viewer.  Also
    # get thumbnails that are lower resolution.
    if args.viewer_cutouts:
        thumbsize = 100
        sample = fits.getdata(samplefile, 1)
        for gal in sample:
            galaxy = gal['GALAXY'].strip().lower()
            size = np.ceil(10*gal['RADIUS']/PIXSCALE)
            thumbpixscale = PIXSCALE*size/thumbsize

            #imageurl = 'http://legacysurvey.org/viewer/jpeg-cutout-decals-dr2?ra={:.6f}&dec={:.6f}'.format(gal['RA'], gal['DEC'])+\
            #  '&pixscale={:.3f}&size={:g}'.format(PIXSCALE, size)
            #imagejpg = os.path.join(largedir, 'cutouts', '{}-image.jpg'.format(galaxy))
            #if os.path.isfile(imagejpg):
            #    os.remove(imagejpg)
            #os.system('wget --continue -O {:s} "{:s}"' .format(imagejpg, imageurl))

            thumburl = 'http://legacysurvey.org/viewer/jpeg-cutout-decals-dr2?ra={:.6f}&dec={:.6f}'.format(gal['RA'], gal['DEC'])+\
              '&pixscale={:.3f}&size={:g}'.format(thumbpixscale, thumbsize)
            thumbjpg = os.path.join(largedir, 'cutouts', '{}-image-thumb.jpg'.format(galaxy))
            if os.path.isfile(thumbjpg):
                os.remove(thumbjpg)
            os.system('wget --continue -O {:s} "{:s}"' .format(thumbjpg, thumburl))

    # --------------------------------------------------
    # Get cutouts of all the CCDs for each galaxy.
    if args.ccd_cutouts:
        sample = fits.getdata(samplefile, 1)

        for gal in sample[1:2]:
            galaxy = gal['GALAXY'].strip().lower()
            ccdsfile = os.path.join(largedir, 'ccds', '{}-ccds.fits'.format(galaxy))
            ccds = fits.getdata(ccdsfile)

            pdb.set_trace()

    # --------------------------------------------------
    # Run the pipeline.
    if args.runbrick:
        sample = fits.getdata(samplefile, 1)

        for gal in sample[1:2]:
            galaxy = gal['GALAXY'].strip().lower()
            diam = 10*np.ceil(gal['RADIUS']/PIXSCALE).astype('int16') # [pixels]

            # Note: zoom is relative to the center of an imaginary brick with
            # dimensions (0, 3600, 0, 3600).
            # blobxy = zip([1800], [1800])
            survey = LegacySurveyData(version='dr2', output_dir=largedir)
            run_brick(None, survey, radec=(gal['RA'], gal['DEC']), blobxy=None, 
                      threads=10, zoom=(1800-diam/2, 1800+diam/2, 1800-diam/2, 1800+diam/2),
                      wise=False, forceAll=True, writePickles=False, do_calibs=False,
                      write_metrics=False, pixPsf=True, splinesky=True, 
                      early_coadds=True, stages=['writecat'], ceres=False)

            pdb.set_trace()

    # --------------------------------------------------
    # Annotate the image/model/resid jpg cutout after running the custom pipeline.
    if args.runbrick_cutouts:
        sample = fits.getdata(samplefile, 1)

        for gal in sample[1:2]:
            galaxy = gal['GALAXY'].strip().lower()
            ra = gal['RA']
            dec = gal['DEC']
            brick = 'custom-{:06d}{}{:05d}'.format(int(1000*ra), 'm' if dec < 0 else 'p',
                                                   int(1000*np.abs(dec)))
            tractorfile = os.path.join(largedir, 'tractor', 'cus', 'tractor-{}.fits'.format(brick))
            print('Reading {}'.format(tractorfile))
            cat = fits.getdata(tractorfile, 1)

            rad = 10
            for imtype in ('image', 'model', 'resid'):
                cutoutfile = os.path.join(largedir, 'cutouts', '{}-runbrick-{}.jpg'.format(galaxy, imtype))
                imfile = os.path.join(largedir, 'coadd', 'cus', brick, 'legacysurvey-{}-{}.jpg'.format(brick, imtype))
                print('Reading {}'.format(imfile))

                im = Image.open(imfile)
                sz = im.size
                draw = ImageDraw.Draw(im)
                for thistype, thiscolor in zip(objtype, objcolor):
                    these = np.where(cat['TYPE'].strip().upper() == thistype)[0]
                    if len(these) > 0:
                        [draw.ellipse((obj['BX']-rad, sz[1]-obj['BY']-rad, obj['BX']+rad,
                                       sz[1]-obj['BY']+rad), outline=thiscolor) for obj in cat[these]]
                # Add a legend.
                im.save(cutoutfile)
                
            pdb.set_trace()

    # --------------------------------------------------
    # Build the webpage.
    if args.build_webpage:

        # index.html
        html = open(os.path.join(largedir, 'index.html'), 'w')
        html.write('<html><body>\n')
        html.write('<h1>Sample of Large Galaxies</h1>\n')
        html.write('<table border="2" width="30%"><tbody>\n')
        sample = fits.getdata(samplefile, 1)
        for gal in sample:
            # Add coordinates and sizes here.
            galaxy = gal['GALAXY'].strip().lower()
            html.write('<tr>\n')
            html.write('<td><a href="html/{}.html">{}</a></td>\n'.format(galaxy, galaxy.upper()))
            html.write('<td><a href="http://legacysurvey.org/viewer/?ra={:.6f}&dec={:.6f}" target="_blank"><img src=cutouts/{}-image-thumb.jpg alt={} /></a></td>\n'.format(gal['RA'], gal['DEC'], galaxy, galaxy.upper()))
#           html.write('<td><a href="html/{}.html"><img src=cutouts/{}-image-thumb.jpg alt={} /></a></td>\n'.format(galaxy, galaxy, galaxy.upper()))
            html.write('</tr>\n')
        html.write('</tbody></table>\n')
        html.write('</body></html>\n')
        html.close()

        # individual galaxy pages
        for gal in sample[:3]:
            galaxy = gal['GALAXY'].strip().lower()
            html = open(os.path.join(largedir, 'html/{}.html'.format(galaxy)), 'w')
            html.write('<html>\n')
            html.write('<head>\n')
            html.write('<style type="text/css">\n')
            html.write('table {width: 90%; }\n')
            html.write('table, th, td {border: 1px solid black; }\n')
            html.write('img {width: 100%; }\n')
            #html.write('td {width: 100px; }\n')
            #html.write('h2 {color: orange; }\n')
            html.write('</style>\n')
            html.write('</head>\n')
            html.write('<body>\n')
            html.write('<h1>{}</h1>\n'.format(galaxy.upper()))
            # ----------
            # Pipeline cutouts
            html.write('<h2>DR2 Pipeline</h2>\n')
            html.write('<table><tbody>\n')
            html.write('<tr>\n')
            #for imtype in ('image', 'model', 'resid'):
            for imtype in ('image', 'image', 'image'): # Hack!
                html.write('<td><a href=../cutouts/{}-{}.jpg><img src=../cutouts/{}-{}.jpg alt={} /></a></td>\n'.format(
                    galaxy, imtype, galaxy, imtype, galaxy.upper()))
            html.write('</tr>\n')
            html.write('</tbody></table>\n')
            # ----------
            # Updated cutouts
            html.write('<h2>Large-Galaxy Pipeline</h2>\n')
            html.write('<table><tbody>\n')
            html.write('<tr>\n')
            for imtype in ('image', 'model', 'resid'):
                html.write('<td><a href=../cutouts/{}-runbrick-{}.jpg><img width="100%" src=../cutouts/{}-runbrick-{}.jpg alt={} /></a></td>\n'.format(
                    galaxy, imtype, galaxy, imtype, galaxy.upper()))
            html.write('</tr>\n')
            html.write('</tbody></table>\n')

            html.write('</body></html>\n')
            html.close()
            
if __name__ == "__main__":
    main()
