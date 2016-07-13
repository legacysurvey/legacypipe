#!/usr/bin/env python

"""This is a temporary hack script to repackage the NAOC-reduced BASS
data into more compact multi-extension FITS files and with a slightly
simplified naming scheme.

The output files are currently written to my scratch directory on edison:
  /scratch2/scratchdirs/ioannis/bok-reduced

TODO (@moustakas):
* Clean up the primary header (right now it's just a copy of the CCD1
  header but with EXTNAME removed).
* Consider compressing the output FITS files (e.g., fz).
* Should we generate weight maps???  What about bad pixel masks?
* Figure out why the headers give CTYPE[1,2] as RA---TAN,DEC--TAN even
  though there are PV distortion coefficients.
* The gain and readnoise for each CCD/amplifier should be in the
  header (and measured frequently!)
* All the images should have consistent units (electron, to be
  consistent with the CP-processed images).  Some are in ADU and
  calibrated using SDSS/UCAC4 and some are in electron(?) (calibrated
  on PS1).
* There are at least three truncated files in the processed data.
* Some supposedly calibrated images are missing the CALI_REF header
  card, including most of the files in 20160209.  A QA code should be
  run on all the processed data after it has been rsynced to NERSC.
* The astrometry on some of these fields is clearly failing, e.g.,
  p_74010159_g.fits
* We should globally replace the filter name 'bokr' with just 'r'.
* Make sure EXTNAME exists for every header.

J. Moustakas
Siena College
2016 July 6

"""
from __future__ import division, print_function

import os
import argparse

import numpy as np
from glob import glob

from astropy.io import fits

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', type=str, help='Desired output directory.')
    args = parser.parse_args()

    # export BASSDATA=/global/project/projectdirs/cosmo/staging/bok
    key = 'BASSDATA'
    if key not in os.environ:
        print('Required ${} environment variable not set'.format(key))
        return 0
    bassdata_dir = os.getenv(key)

    if args.output_dir is None:
        output_dir = os.path.join(os.getenv('SCRATCH'), 'bok-reduced')
    else:
        output_dir = args.output_dir
        
    try:
        os.stat(output_dir)
        print('Writing to top-level directory {}'.format(output_dir))
    except:
        print('Creating top-level directory {}'.format(output_dir))
        os.mkdir(output_dir)
        
    # If only one filter was used during the night then the directory
    # doesn't have a trailing filter name.
    dirs = np.concatenate((
            glob(os.path.join(bassdata_dir, 'reduced', '201[5-6]????')),
            glob(os.path.join(bassdata_dir, 'reduced', '201[5-6]?????'))
            ))
    dirs = dirs[np.argsort(dirs)]
    #dirs = dirs[19:20]
    #for thisdir in dirs:
    #    print(thisdir)
    #import pdb ; pdb.set_trace()

    # These files are corrupted/incomplete.
    badfiles = ','.join([
            'p7400g0275_1.fits',    # truncated data
            'p7400g0280_1.fits',    # truncated data
            'p7402bokr0044_1.fits', # truncated data
            'p7430bokr0262_1.fits'])

    # List of extensions and gains.  These are the average, not
    # per-amplifier gains, so correcting for gain variations is just a
    # temporary hack.
    extlist = ('_1', '_2', '_3', '_4')
    gainlist = (1.47, 1.48, 1.42, 1.4275)
    
    for thisdir in dirs:
        allfiles = np.array(glob(os.path.join(thisdir, 'p*_1.fits')))
        print('Found {} exposures in directory {}'.format(len(allfiles), thisdir))
        if len(allfiles) > 0: # some directories are empty
            allfiles = allfiles[np.argsort(allfiles)]
            # Remove bad exposures/files.
            isbad = []
            for ii in range(len(allfiles)):
                if os.path.basename(allfiles[ii]) in badfiles:
                    isbad.append(ii)
            allfiles = np.delete(np.array(allfiles), isbad)
            
            for thisfile in allfiles:
                basefile = os.path.basename(thisfile)
                #import pdb ; pdb.set_trace()                
                if 'bokr' in basefile or 'g' in basefile: # not sure if there are other filters
                    if 'bokr' in basefile:
                        filter = 'bokr'
                        outfilter = 'r'
                    else:
                        filter = 'g'
                        outfilter = 'g'
                    expnum = '{}{}'.format(basefile[1:5], basefile[5+len(filter):-7])
                    outdir = os.path.join(output_dir, os.path.basename(thisdir)[:8])
                    try:
                        os.stat(outdir)
                    except:
                        print('Creating output directory {}'.format(outdir))
                        os.mkdir(outdir)
                    outfile = os.path.join(outdir, 'p_{}_{}.fits'.format(expnum, outfilter))
                    if os.path.isfile(outfile):
                        os.remove(outfile)

                    # Build the output file.  In some directories
                    # (e.g., 20160211) it looks like the rsync
                    # transfer was truncated, so check to make sure we
                    # have everything.  Also make sure CALI_REF exists
                    # in the header and replace the FILTER header
                    # card.
                    allhere = True
                    for ext in extlist:
                        thisfile1 = thisfile.replace('_1', ext)
                        if not os.path.isfile(thisfile1):
                            allhere = False
                        hdr = fits.getheader(thisfile1)
                        if not 'CALI_REF' in hdr:
                            allhere = False
                        
                    if allhere:
                        # Temporary hack
                        #if 'PS1' in fits.getheader(thisfile)['CALI_REF']:
                        #    continue
                        hduout = fits.HDUList()
                        for ext, gain in zip(extlist, gainlist):
                            hduin = fits.open(thisfile.replace('_1', ext))
                            # Images calibrated on PS1 are in
                            # electrons, otherwise they're in ADU; fix
                            # that here.
                            hdr, img = hduin[0].header, hduin[0].data
                            hdr['FILTER'] = outfilter
                            if not 'PS1' in hdr['CALI_REF']:
                                img = img * gain 
                            #import pdb ; pdb.set_trace()
                            if ext == '_1':
                                primhdr = hdr.copy()
                                primhdr['EXTNAME'] = 'PRIMARY'
                                hduout.append(fits.PrimaryHDU(header=primhdr))
                            hduout.append(fits.ImageHDU(img, header=hdr))
                            hduin.close()
                        print('  Writing {}'.format(outfile))
                        hduout.writeto(outfile)
                    else:
                        print('Missing some files or metadata for exposure {}'.format(thisfile))
                    
if __name__ == "__main__":
    main()
