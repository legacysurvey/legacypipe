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
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
from astropy.table import Table, vstack
from astropy.visualization import scale_image

from PIL import Image, ImageDraw, ImageFont

from astrometry.util.util import Tan
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import dimshow

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

def _simplewcs(gal):
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
    fonttype = '/usr/share/fonts/gnu-free/FreeSans.ttf'

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
        outcat.write(samplefile)
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

            tims = []
            for ii, ccd in enumerate(ccds):
                im = survey.get_image_object(ccd)
                print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
                      'seeing {:.2f}'.format(ccd.fwhm*im.pixscale), 
                      'object', getattr(ccd, 'object', None))
                tim = im.get_tractor_image(splinesky=True, subsky=False, gaussPsf=True)
                #tims.append(tim)

                # Get the image and read and instantiate the splinesky model.
                image = tim.getImage()
                sky = tim.getSky()
                skymodel = np.zeros_like(image)
                sky.addTo(skymodel)

                qaccd = os.path.join(qadir, 'qa-{}-ccd{:02d}.png'.format(galaxy, ii))
                fig, ax = plt.subplots(1, 2, figsize=(6, 6), sharey=True)
                fig.suptitle(tim.name, y=0.94)
                #vmin, vmax = (0, np.percentile(image, 95))
                #vmin, vmax = np.percentile(image, (5, 95))
                for data, thisax, title in zip((image, skymodel), ax, ('Image', 'SplineSky')):
                    vmin, vmax = np.percentile(data, (5, 98))
                    img = scale_image(data, scale='sqrt', min_cut=vmin, max_cut=vmax)
                    #img = scale_image(data, scale='log', min_percent=0.02, max_percent=0.98)
                    thisim = thisax.imshow(img, cmap='viridis', interpolation='nearest', origin='lower')
                    #thisim = thisax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax,
                    #                       interpolation='nearest', origin='lower')
                    div = make_axes_locatable(thisax)
                    cax = div.append_axes('right', size='15%', pad=0.05)
                    cbar = fig.colorbar(thisim, cax=cax, format='%.3g')
                    #cbar.set_label('Colorbar {}'.format(i), size=10)
                    thisax.set_title(title)
                    thisax.xaxis.set_visible(False)
                    thisax.yaxis.set_visible(False)
                    
                ## Shared colorbar.
                #cbarax = fig.add_axes([0.83, 0.15, 0.03, 0.8])
                #cbar = fig.colorbar(thisim, cax=cbarax)
                plt.tight_layout(w_pad=0.25)
                #plt.subplots_adjust(top=0.85)
                print('Writing {}'.format(qaccd))
                plt.savefig(qaccd)

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
                ax.set_xlabel('Pixel')
                ax.set_ylabel('Pixel')
                ax.set_title('Blobs')
                
            print('Writing {}'.format(qablobsfile))
            plt.savefig(qablobsfile)

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
            html.write('<h2>DR2 Pipeline</h2>\n')
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
            html.write('<h2>Large-Galaxy Pipeline</h2>\n')
            html.write('<table><tbody>\n')
            html.write('<tr>\n')
            for imtype in ('image', 'model', 'resid'):
                html.write('<td><a href=../cutouts/{}-{}-custom-annot.jpg><img width="100%" src=../cutouts/{}-{}-custom-annot.jpg alt={} /></a></td>\n'.format(
                    galaxy, imtype, galaxy, imtype, galaxy.upper()))
            html.write('</tr>\n')
            # ----------
            # Blob diagnostic plot
            html.write('<tr><td><a href=../qa/{}/qa-{}-blobs.png>'.format(galaxy, galaxy)+\
                       '<img src=../qa/{}/qa-{}-blobs.png alt={} Blobs /></a></td>'.format(galaxy, galaxy, galaxy.upper())+\
                       '</tr>\n')
                       #'<td>&nbsp</td><td>&nbsp</td></tr>\n')
            # ----------
            # CCD cutouts
            qadir = os.path.join(largedir, 'qa', '{}'.format(galaxy))
            qaccd = glob(os.path.join(qadir, 'qa-{}-ccd??.png'.format(galaxy)))
            pdb.set_trace()

            for ccd in qaccd:
                html.write('<tr><td><a href=../qa/{}/qa-{}-blobs.png>'.format(galaxy, galaxy)+\
                           '<img src=../qa/{}/qa-{}-blobs.png alt={} Blobs /></a></td>'.format(galaxy, galaxy, galaxy.upper())+\
                           '</tr>\n')
                       #'<td>&nbsp</td><td>&nbsp</td></tr>\n')
            html.write('</tbody></table>\n')
            html.write('</body></html>\n')
            html.close()
            
if __name__ == "__main__":
    main()
