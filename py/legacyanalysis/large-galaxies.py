#!/usr/bin/env python

"""Redo the Tractor photometry of the "large" galaxies in Legacy Survey imaging.

rsync -avPn --files-from='/tmp/ccdfiles.txt' nyx:/usr/local/legacysurvey/legacypipe-dir/ /Users/ioannis/repos/git/legacysurvey/legacypipe-dir/

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
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.ndimage.morphology import binary_dilation

from astropy.io import fits
from astropy.table import Table, vstack
from astropy.visualization import scale_image

from PIL import Image, ImageDraw, ImageFont

from astrometry.util.util import Tan
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import dimshow

from tractor.splinesky import SplineSky

from legacypipe.runbrick import run_brick
from legacypipe.common import ccds_touching_wcs, bricks_touching_wcs, LegacySurveyData

PIXSCALE = 0.262 # average pixel scale [arcsec/pix]
DIAMFACTOR = 10

def _uniqccds(ccds):
    '''Get the unique set of CCD files.'''
    ccdfile = []
    [ccdfile.append('{}-{}'.format(expnum, ccdname)) for expnum,
     ccdname in zip(ccds.expnum, ccds.ccdname)]
    _, indx = np.unique(ccdfile, return_index=True)
    return indx

def _getfiles(ccdinfo):
    '''Figure out the set of images and calibration files we need to transfer, if any.'''

    nccd = len(ccdinfo)

    survey = LegacySurveyData()
    calibdir = survey.get_calib_dir()
    imagedir = survey.survey_dir

    expnum = ccdinfo.expnum
    ccdname = ccdinfo.ccdname

    psffiles = list()
    skyfiles = list()
    imagefiles = list()
    for ccd in ccdinfo:
        info = survey.get_image_object(ccd)
        for attr in ['imgfn', 'dqfn', 'wtfn']:
            imagefiles.append(getattr(info, attr).replace(imagedir+'/', ''))
        psffiles.append(info.psffn.replace(calibdir, 'calib'))
        skyfiles.append(info.splineskyfn.replace(calibdir, 'calib'))

    ccdfiles = open('/tmp/ccdfiles.txt', 'w')
    for ff in psffiles:
        ccdfiles.write(ff+'\n')
    for ff in skyfiles:
        ccdfiles.write(ff+'\n')
    for ff in imagefiles:
        ccdfiles.write(ff+'\n')
    ccdfiles.close()

    cmd = "rsync -avPn --files-from='/tmp/ccdfiles.txt' edison:/global/cscratch1/sd/desiproc/dr3/ /global/work/decam/versions/work/"
    print('You should run the following command:')
    print('  {}'.format(cmd))

def _catalog_template(nobj=1):
    from astropy.table import Table
    cols = [
        ('GALAXY', 'S20'), 
        ('RA', 'f8'), 
        ('DEC', 'f8'),
        ('RADIUS', 'f4'),
        ('BRICKNAME', 'S17', (4,))
        ]
    catalog = Table(np.zeros(nobj, dtype=cols))
    catalog['RADIUS'].unit = 'arcsec'

    return catalog

def _ccdwcs(ccd):
    '''Build a simple WCS object for each CCD.'''
    W, H = ccd.width, ccd.height
    ccdwcs = Tan(*[float(xx) for xx in [ccd.crval1, ccd.crval2, ccd.crpix1,
                                        ccd.crpix2, ccd.cd1_1, ccd.cd1_2,
                                        ccd.cd2_1, ccd.cd2_2, W, H]])
    return ccdwcs

def _galwcs(gal):
    '''Build a simple WCS object for a single galaxy.'''
    diam = DIAMFACTOR*np.ceil(gal['RADIUS']/PIXSCALE).astype('int16') # [pixels]
    galwcs = Tan(gal['RA'], gal['DEC'], diam/2+0.5, diam/2+0.5,
                 -PIXSCALE/3600.0, 0.0, 0.0, PIXSCALE/3600.0, 
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

    key = 'DECALS_DR2_DIR'
    if key not in os.environ:
        print('Required ${} environment variable not set'.format(key))
        return 0
    drdir = os.getenv(key)

    # Some convenience variables.
    objtype = ('PSF', 'SIMP', 'EXP', 'DEV', 'COMP')
    objcolor = ('white', 'red', 'orange', 'cyan', 'yellow')
    thumbsize = 100
    #fonttype = '/usr/share/fonts/gnu-free/FreeSans.ttf'
    fonttype = '/Volumes/Macintosh\ HD/Library/Fonts/Georgia.ttf'

    # Read the sample (unless we're building it!)
    samplefile = os.path.join(largedir, 'large-galaxies-sample.fits')
    if not args.build_sample:
        sample = fits.getdata(samplefile, 1)
        sample = sample[2:3] # Hack!
        #sample = sample[np.where(np.sum((sample['BRICKNAME'] != '')*1, 1) > 1)[0]]
        #pdb.set_trace()

    survey = LegacySurveyData(version='dr2') # update to DR3!
      
    # --------------------------------------------------
    # Build the sample of large galaxies based on the available imaging.
    if args.build_sample:
        # Read the parent catalog.
        cat = read_rc3()
        cat = cat[np.where(cat['GALAXY'] == 'UGC4203')[0]]
        
        # Create a simple WCS object for each object and find all the CCDs
        # touching that WCS footprint.
        bricks = survey.get_bricks_dr2()

        allccds = survey.get_ccds()
        keep = np.concatenate((survey.apply_blacklist(allccds),
                               survey.photometric_ccds(allccds)))
        allccds.cut(keep)

        ccdlist = []
        outcat = []
        for gal in cat:
            galwcs = _galwcs(gal)

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

                # Get the brickname
                rad = 2*gal['RADIUS']/3600 # [degree]
                brickindx = survey.bricks_touching_radec_box(bricks,
                                                             gal['RA']-rad, gal['RA']+rad,
                                                             gal['DEC']-rad, gal['DEC']+rad)
                if len(brickindx) == 0 or len(brickindx) > 4:
                    print('This should not happen!')
                    pdb.set_trace()
                gal['BRICKNAME'][:len(brickindx)] = bricks.brickname[brickindx]
                #gal['BRICKNAME'] = ' '.join(bricks.brickname[brickindx])

                if len(outcat) == 0:
                    outcat = gal
                else:
                    outcat = vstack((outcat, gal))

        # Write out the final catalog.
        samplefile = os.path.join(largedir, 'large-galaxies-sample.fits')
        if os.path.isfile(samplefile):
            os.remove(samplefile)
        print('Writing {}'.format(samplefile))
        #outcat.write(samplefile)
        print(outcat)

        # Do we need to transfer any of the data to nyx?
        _getfiles(merge_tables(ccdlist))

    # --------------------------------------------------
    # Get data, model, and residual cutouts from the legacysurvey viewer.  Also
    # get thumbnails that are lower resolution.
    if args.viewer_cutouts:
        for gal in sample:
            galaxy = gal['GALAXY'].strip().lower()

            # SIZE here should be consistent with DIAM in args.runbrick, below
            size = DIAMFACTOR*np.ceil(gal['RADIUS']/PIXSCALE).astype('int16') # [pixels]
            thumbpixscale = PIXSCALE*size/thumbsize

            # Get cutouts of the data, model, and residual images.
            for imtype, tag in zip(('image', 'model', 'resid'), ('', '&tag=decals-model', '&tag=decals-resid')):
                imageurl = 'http://legacysurvey.org/viewer-dev/jpeg-cutout-decals-dr2?ra={:.6f}&dec={:.6f}&pixscale={:.3f}&size={:g}{}'.format(gal['RA'], gal['DEC'], PIXSCALE, size, tag)
                imagejpg = os.path.join(largedir, 'cutouts', '{}-{}.jpg'.format(galaxy, imtype))
                print('Uncomment me to redownload')
                #if os.path.isfile(imagejpg):
                #    os.remove(imagejpg)
                #os.system('wget --continue -O {:s} "{:s}"' .format(imagejpg, imageurl))

            # Also get a small thumbnail of just the image.
            thumburl = 'http://legacysurvey.org/viewer-dev/jpeg-cutout-decals-dr2?ra={:.6f}&dec={:.6f}'.format(gal['RA'], gal['DEC'])+\
              '&pixscale={:.3f}&size={:g}'.format(thumbpixscale, thumbsize)
            thumbjpg = os.path.join(largedir, 'cutouts', '{}-image-thumb.jpg'.format(galaxy))
            #if os.path.isfile(thumbjpg):
            #    os.remove(thumbjpg)
            #os.system('wget --continue -O {:s} "{:s}"' .format(thumbjpg, thumburl))

            rad = 10
            for imtype in ('image', 'model', 'resid'):
                imfile = os.path.join(largedir, 'cutouts', '{}-{}.jpg'.format(galaxy, imtype))
                cutoutfile = os.path.join(largedir, 'cutouts', '{}-{}-runbrick-annot.jpg'.format(galaxy, imtype))

                print('Reading {}'.format(imfile))
                im = Image.open(imfile)
                sz = im.size
                fntsize = np.round(sz[0]/35).astype('int')
                font = ImageFont.truetype(fonttype, size=fntsize)
                # Annotate the sources on each cutout.
                wcscutout = Tan(gal['RA'], gal['DEC'], sz[0]/2+0.5, sz[1]/2+0.5,
                                -PIXSCALE/3600.0, 0.0, 0.0, PIXSCALE/3600.0, float(sz[0]), float(sz[1]))
                draw = ImageDraw.Draw(im)
                for bb, brick in enumerate(gal['BRICKNAME'][np.where(gal['BRICKNAME'] != '')[0]]):
                    tractorfile = os.path.join(drdir, 'tractor', '{}'.format(brick[:3]),
                                               'tractor-{}.fits'.format(brick))
                    print('  Reading {}'.format(tractorfile))
                    cat = fits.getdata(tractorfile, 1)
                    cat = cat[np.where(cat['BRICK_PRIMARY']*1)[0]]
                    for ii, (thistype, thiscolor) in enumerate(zip(objtype, objcolor)):
                        these = np.where(cat['TYPE'].strip().upper() == thistype)[0]
                        if len(these) > 0:
                            for obj in cat[these]:
                                ok, xx, yy = wcscutout.radec2pixelxy(obj['RA'], obj['DEC'])
                                xx -= 1 # PIL is zero-indexed
                                yy -= 1
                                #print(obj['RA'], obj['DEC'], xx, yy)
                                draw.ellipse((xx-rad, sz[1]-yy-rad, xx+rad, sz[1]-yy+rad),
                                             outline=thiscolor)

                        # Add a legend, but just after the first brick.
                        if bb == 0:
                            draw.text((20, 20+ii*fntsize*1.2), thistype, font=font, fill=thiscolor)
                draw.text((sz[0]-fntsize*4, sz[1]-fntsize*2), imtype.upper(), font=font)
                print('Writing {}'.format(cutoutfile))
                im.save(cutoutfile)
                #pdb.set_trace()

    # --------------------------------------------------
    # Get cutouts and build diagnostic plots of all the CCDs for each galaxy.
    if args.ccd_cutouts:
        for gal in sample:
            galaxy = gal['GALAXY'].strip().lower()
            print('Building CCD QA for galaxy {}'.format(galaxy.upper()))
            qadir = os.path.join(largedir, 'qa', '{}'.format(galaxy))
            try:
                os.stat(qadir)
            except:
                os.mkdir(qadir)
            
            ccdsfile = os.path.join(largedir, 'ccds', '{}-ccds.fits'.format(galaxy))
            ccds = fits_table(ccdsfile)
            #print('Hack!!!  Testing with 3 CCDs!')
            #ccds = ccds[4:5]

            # Build a QAplot showing the position of all the CCDs and the coadd
            # cutout (centered on the galaxy).
            galwcs = _galwcs(gal)
            pxscale = galwcs.pixel_scale()/3600.0
            (width, height) = (galwcs.get_width()*pxscale, galwcs.get_height()*pxscale)
            bb = galwcs.radec_bounds()
            bbcc = galwcs.radec_center()
            ww = 0.2

            qaccdposfile = os.path.join(qadir, 'qa-{}-ccdpos.png'.format(galaxy))
            fig, allax = plt.subplots(1, 3, figsize=(12, 5), sharey=True, sharex=True)

            for ax, band in zip(allax, ('g', 'r', 'z')):
                ax.set_aspect('equal')
                ax.set_xlim(bb[0]+width+ww, bb[0]-ww)
                ax.set_ylim(bb[2]-ww, bb[2]+height+ww)
                ax.set_xlabel('RA (deg)')
                ax.text(0.9, 0.05, band, ha='center', va='bottom',
                        transform=ax.transAxes, fontsize=18)

                if band == 'g':
                    ax.set_ylabel('Dec (deg)')
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                ax.add_patch(patches.Rectangle((bb[0], bb[2]), bb[1]-bb[0], bb[3]-bb[2],
                                               fill=False, edgecolor='black', lw=3, ls='--'))
                ax.add_patch(patches.Circle((bbcc[0], bbcc[1]), gal['RADIUS']/3600.0/2, 
                                            fill=False, edgecolor='black', lw=2))

                these = np.where(ccds.filter == band)[0]
                col = plt.cm.Set1(np.linspace(0, 1, len(ccds)))
                for ii, ccd in enumerate(ccds[these]):
                    print(ccd.expnum, ccd.ccdname, ccd.filter)
                    W, H = ccd.width, ccd.height
                    ccdwcs = _ccdwcs(ccd)

                    cc = ccdwcs.radec_bounds()
                    ax.add_patch(patches.Rectangle((cc[0], cc[2]), cc[1]-cc[0],
                                                   cc[3]-cc[2], fill=False, lw=2, 
                                                   edgecolor=col[these[ii]],
                                                   label='ccd{:02d}'.format(these[ii])))
                    ax.legend(ncol=2, frameon=False, loc='upper left')
                    
            plt.subplots_adjust(bottom=0.12, wspace=0.05, left=0.06, right=0.97)
            print('Writing {}'.format(qaccdposfile))
            plt.savefig(qaccdposfile)

            #print('Exiting prematurely!')
            #sys.exit(1)

            #tims = []
            for ii, ccd in enumerate(ccds):
                im = survey.get_image_object(ccd)
                print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
                      'seeing {:.2f}'.format(ccd.fwhm*im.pixscale), 
                      'object', getattr(ccd, 'object', None))
                tim = im.get_tractor_image(splinesky=True, subsky=False, gaussPsf=True)
                #tims.append(tim)

                # Get the (pixel) coordinates of the galaxy on this CCD
                ok, x0, y0 = tim.wcs.radec2pixelxy(gal['RA'], gal['DEC'])
                pxscale = wcs.pixel_scale()
                radius = DIAMFACTOR*gal['RADIUS']/pxscale/2.0

                # Get the image, read and instantiate the splinesky model, and
                # also reproduce the image mask used in legacypipe.decam.run_calibs.
                image = tim.getImage()
                weight = tim.getInvvar()
                sky = tim.getSky()
                skymodel = np.zeros_like(image)
                sky.addTo(skymodel)

                med = np.median(image[weight > 0])
                skyobj = SplineSky.BlantonMethod(image - med, weight>0, 512)
                skymod = np.zeros_like(image)
                skyobj.addTo(skymod)
                sig1 = 1.0/np.sqrt(np.median(weight[weight > 0]))
                mask = ((image - med - skymod) > (5.0*sig1))*1.0
                mask = binary_dilation(mask, iterations=3)
                mask[weight == 0] = 1

                #plt.clf()
                #plt.figure()
                #plt.hist(image-skymodel, range=(0, 0.1), bins=10) ; plt.show()
                #plt.savefig('junk.png')

                qaccd = os.path.join(qadir, 'qa-{}-ccd{:02d}.png'.format(galaxy, ii))
                fig, ax = plt.subplots(1, 3, figsize=(9, 5), sharey=True)
                fig.suptitle('{} (ccd{:02d})'.format(tim.name, ii), y=0.96)
                #vmin, vmax = (0, np.percentile(image, 95))
                #vmin, vmax = np.percentile(image, (2, 90))
                for data, thisax, title in zip((image, skymodel, mask), ax, ('Image', 'SplineSky', 'Mask')):
                    if title == 'Mask':
                        vmin, vmax = (0, 1)
                    else:
                        vmin, vmax = np.percentile(data, (1, 99))
                    #img = scale_image(data, scale='sqrt', min_cut=vmin, max_cut=vmax)
                    #img = scale_image(data, scale='sqrt')#, min_percent=0.02, max_percent=0.98)
                    thisim = thisax.imshow(data, cmap='inferno', interpolation='nearest', origin='lower',
                                           vmin=vmin, vmax=vmax)
                    thisax.add_patch(patches.Circle((x0, y0), radius, fill=False, edgecolor='white', lw=2))

                    #thisim = thisax.imshow(data, cmap='gray', interpolation='nearest', origin='lower',
                    #                       norm=LogNorm())
                    #thisim = thisax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax,
                    #                       interpolation='nearest', origin='lower')
                    #thisim = dimshow(data, vmin=-3.0*tim.sig1, vmax=10.0*tim.sig1)
                    #if title != 'Mask':
                    div = make_axes_locatable(thisax)
                    cax = div.append_axes('right', size='15%', pad=0.1)
                    cbar = fig.colorbar(thisim, cax=cax, format='%.4g')
                        #cbar.set_label('Colorbar {}'.format(i), size=10)
                    thisax.set_title(title)
                    thisax.xaxis.set_visible(False)
                    thisax.yaxis.set_visible(False)
                    
                ## Shared colorbar.
                #cbarax = fig.add_axes([0.83, 0.15, 0.03, 0.8])
                #cbar = fig.colorbar(thisim, cax=cbarax)
                plt.tight_layout(w_pad=0.25)
                plt.subplots_adjust(bottom=0.0, top=0.93)
                print('Writing {}'.format(qaccd))
                plt.savefig(qaccd)

                #print('Exiting prematurely!')
                #sys.exit(1)
                #sys.exit(1)
                #pdb.set_trace()

    # --------------------------------------------------
    # Run the pipeline.
    if args.runbrick:
        for gal in sample:
            galaxy = gal['GALAXY'].strip().lower()

            # DIAM here should be consistent with SIZE in args.viewer_cutouts,
            # above.  Also note that ZOOM is relative to the center of an
            # imaginary brick with dimensions (0, 3600, 0, 3600).
            diam = DIAMFACTOR*np.ceil(gal['RADIUS']/PIXSCALE).astype('int16') # [pixels]
            zoom = (1800-diam/2, 1800+diam/2, 1800-diam/2, 1800+diam/2)

            nsigma = 20
            #blobxy = zip([1800], [1800])
            survey = LegacySurveyData(version='dr2', output_dir=largedir)
            run_brick(None, survey, radec=(gal['RA'], gal['DEC']), blobxy=None, 
                      threads=10, zoom=zoom, wise=False, forceAll=True, writePickles=False,
                      do_calibs=False, write_metrics=True, pixPsf=True, splinesky=True, 
                      early_coadds=False, stages=['writecat'], ceres=False, nsigma=nsigma,
                      plots=True)

            #pdb.set_trace()

    # --------------------------------------------------
    # Annotate the image/model/resid jpg cutouts after running the custom pipeline.
    if args.runbrick_cutouts:
        for gal in sample:
            galaxy = gal['GALAXY'].strip().lower()
            qadir = os.path.join(largedir, 'qa', '{}'.format(galaxy))

            ra = gal['RA']
            dec = gal['DEC']
            brick = 'custom-{:06d}{}{:05d}'.format(int(1000*ra), 'm' if dec < 0 else 'p',
                                                   int(1000*np.abs(dec)))
            tractorfile = os.path.join(largedir, 'tractor', 'cus', 'tractor-{}.fits'.format(brick))
            blobsfile = os.path.join(largedir, 'metrics', 'cus', '{}'.format(brick), 'blobs-{}.fits.gz'.format(brick))

            print('Reading {}'.format(tractorfile))
            cat = fits.getdata(tractorfile, 1)

            print('Reading {}'.format(blobsfile))
            blobs = fits.getdata(blobsfile)

            qablobsfile = os.path.join(qadir, 'qa-{}-blobs.png'.format(galaxy))
            fig, ax = plt.subplots(1, figsize=(6, 6))
            dimshow(blobs != -1)
            for blob in np.unique(cat['BLOB']):
                these = np.where(cat['BLOB'] == blob)[0]
                xx, yy = (np.mean(cat['BX'][these]), np.mean(cat['BY'][these]))
                ax.text(xx, yy, '{}'.format(blob), ha='center', va='bottom', color='orange')
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            
            print('Writing {}'.format(qablobsfile))
            plt.savefig(qablobsfile)

            print('Exiting prematurely!')
            sys.exit(1)

            rad = 10
            for imtype in ('image', 'model', 'resid'):
                cutoutfile = os.path.join(largedir, 'cutouts', '{}-{}-custom-annot.jpg'.format(galaxy, imtype))
                imfile = os.path.join(largedir, 'coadd', 'cus', brick, 'legacysurvey-{}-{}.jpg'.format(brick, imtype))
                print('Reading {}'.format(imfile))

                im = Image.open(imfile)
                sz = im.size
                fntsize = np.round(sz[0]/35).astype('int')
                font = ImageFont.truetype(fonttype, size=fntsize)
                draw = ImageDraw.Draw(im)
                for ii, (thistype, thiscolor) in enumerate(zip(objtype, objcolor)):
                    these = np.where(cat['TYPE'].strip().upper() == thistype)[0]
                    if len(these) > 0:
                        [draw.ellipse((obj['BX']-rad, sz[1]-obj['BY']-rad, obj['BX']+rad,
                                       sz[1]-obj['BY']+rad), outline=thiscolor) for obj in cat[these]]
                    # Add a legend.
                    draw.text((20, 20+ii*fntsize*1.2), thistype, font=font, fill=thiscolor)
                draw.text((sz[0]-fntsize*4, sz[1]-fntsize*2), imtype.upper(), font=font)
                #draw.text((sz[0], sz[1]-20), imtype.upper(), font=font)
                im.save(cutoutfile)
                
    # --------------------------------------------------
    # Build the webpage.
    if args.build_webpage:
        sample = fits.getdata(samplefile, 1)
        
        # index.html
        htmlfile = os.path.join(largedir, 'index.html')
        print('Writing {}'.format(htmlfile))
        html = open(htmlfile, 'w')
        html.write('<html><body>\n')
        html.write('<h1>Sample of Large Galaxies</h1>\n')
        html.write('<table border="2" width="30%"><tbody>\n')
        for ii, gal in enumerate(sample):
            # Add coordinates and sizes here.
            galaxy = gal['GALAXY'].strip().lower()
            html.write('<tr>\n')
            html.write('<td>{}</td>\n'.format(ii))
            html.write('<td><a href="html/{}.html">{}</a></td>\n'.format(galaxy, galaxy.upper()))
            html.write('<td><a href="http://legacysurvey.org/viewer/?ra={:.6f}&dec={:.6f}" target="_blank"><img src=cutouts/{}-image-thumb.jpg alt={} /></a></td>\n'.format(gal['RA'], gal['DEC'], galaxy, galaxy.upper()))
            html.write('</tr>\n')
        html.write('</tbody></table>\n')
        html.write('</body></html>\n')
        html.close()

        # individual galaxy pages
        for gal in sample:
            galaxy = gal['GALAXY'].strip().lower()
            qadir = os.path.join(largedir, 'qa', '{}'.format(galaxy))

            htmlfile = os.path.join(largedir, 'html/{}.html'.format(galaxy))
            print('Writing {}'.format(htmlfile))
            html = open(htmlfile, 'w')
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
            # DR2 Pipeline cutouts
            html.write('<h2>DR2 Pipeline (Image, Model, Residuals)</h2>\n')
            html.write('<table><tbody>\n')
            html.write('<tr>\n')
            for imtype in ('image', 'model', 'resid'):
                html.write('<td><a href=../cutouts/{}-{}-runbrick-annot.jpg>'.format(galaxy, imtype)+\
                           '<img src=../cutouts/{}-{}-runbrick-annot.jpg alt={} /></a></td>\n'.format(galaxy, imtype, galaxy.upper()))
            html.write('</tr>\n')
            #html.write('<tr><td>Data</td><td>Model</td><td>Residuals</td></tr>\n')
            html.write('</tbody></table>\n')
            # ----------
            # Large-Galaxy custom pipeline cutouts
            html.write('<h2>Large-Galaxy Pipeline (Image, Model, Residuals)</h2>\n')
            html.write('<table><tbody>\n')
            html.write('<tr>\n')
            for imtype in ('image', 'model', 'resid'):
                html.write('<td><a href=../cutouts/{}-{}-custom-annot.jpg><img width="100%" src=../cutouts/{}-{}-custom-annot.jpg alt={} /></a></td>\n'.format(
                    galaxy, imtype, galaxy, imtype, galaxy.upper()))
            html.write('</tr>\n')
            html.write('</tbody></table>\n')
            # ----------
            # Blob diagnostic plot
            html.write('<h2>Segmentation (nsigma>20)</h2>\n')
            html.write('<table><tbody>\n')
            html.write('<tr><td><a href=../qa/{}/qa-{}-blobs.png>'.format(galaxy, galaxy)+\
                       '<img src=../qa/{}/qa-{}-blobs.png alt={} Blobs /></a></td>'.format(galaxy, galaxy, galaxy.upper())+\
                       '</tr>\n')
                       #'<td>&nbsp</td><td>&nbsp</td></tr>\n')
            html.write('</tbody></table>\n')
            # ----------
            # CCD cutouts
            html.write('<h2>Configuration of CCDs and Sky-Subtraction</h2>\n')
            html.write('<table><tbody>\n')
            html.write('<tr><td><a href=../qa/{}/qa-{}-ccdpos.png>'.format(galaxy, galaxy)+\
                       '<img src=../qa/{}/qa-{}-ccdpos.png alt={} CCD Positions /></a></td>'.format(galaxy, galaxy, galaxy.upper())+\
                       '</tr>\n')
            html.write('</tbody></table>\n')
            qaccd = glob(os.path.join(qadir, 'qa-{}-ccd??.png'.format(galaxy)))
            if len(qaccd) > 0:
                html.write('<table><tbody>\n')
                for ccd in qaccd:
                    qaccd1 = os.path.split(ccd)[-1]
                    html.write('<tr><td><a href=../qa/{}/{}>'.format(galaxy, qaccd1)+\
                               '<img src=../qa/{}/{} alt={} Sky in CCD /></a></td>'.format(galaxy, qaccd1, galaxy.upper())+\
                               '</tr>\n')
                html.write('</tbody></table>\n')
            html.write('</body></html>\n')
            html.close()
            
if __name__ == "__main__":
    main()
