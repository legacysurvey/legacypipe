#!/usr/bin/env python

"""This is a temporary hack script to repackage the NAOC-reduced BASS
data into more compact multi-extension FITS files and with a slightly
simplified naming scheme.

The output files are currently written to my scratch directory on edison:
  /scratch2/scratchdirs/ioannis/bok-reduced

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
    parser.add_argument('--build-sample', action='store_true', help='Build the sample.')
    args = parser.parse_args()

    # export BASSDATA=/global/project/projectdirs/cosmo/staging/bok
    key = 'BASSDATA'
    if key not in os.environ:
        print('Required ${} environment variable not set'.format(key))
        return 0
    bassdata_dir = os.getenv(key)

    # temporary hack
    output_dir = os.path.join(os.getenv('SCRATCH'), 'bok-reduced')
    try:
        os.stat(output_dir)
    except:
        print('Creating top-level directory {}'.format(output_dir))
        os.mkdir(output_dir)

    # If only one filter was used during the night then the directory
    # doesn't have a trailing filter name.
    dirs1 = glob(os.path.join(bassdata_dir, 'reduced', '201[5-6]????'))
    dirs2 = glob(os.path.join(bassdata_dir, 'reduced', '201[5-6]?????'))
    dirs = np.concatenate((dirs1, dirs2))

    for dir in dirs:
        allfiles = glob(os.path.join(dir, 'p*_1.fits'))
        if len(allfiles) > 0: # some directories are empty
            for thisfile in allfiles:
                basefile = os.path.basename(thisfile)
                if 'bokr' in basefile or 'g' in basefile: # not sure if there are other filters
                    if 'bokr' in basefile:
                        filter = 'bokr'
                    else:
                        filter = 'g'
                    expnum = '{}_{}'.format(basefile[1:5], basefile[5+len(filter):-7])
                    outdir = os.path.join(output_dir, os.path.basename(dir)[:8])
                    try:
                        os.stat(outdir)
                    except:
                        print('Creating output directory {}'.format(outdir))
                        os.mkdir(outdir)
                    outfile = os.path.join(outdir, 'p_{}_{}.fits'.format(expnum, filter))
                    if os.path.isfile(outfile):
                        os.remove(outfile)

                    # Build the output file.
                    hduout = fits.HDUList()
                    hduout.append(fits.PrimaryHDU())
                    for ext in ('_1', '_2', '_3', '_4'):
                        hduin = fits.open(thisfile.replace('_1', ext))
                        hduout.append(fits.ImageHDU(hduin[0].data, header=hduin[0].header))
                        hduin.close()
                    print('  Writing {}'.format(outfile))
                    hduout.writeto(outfile)
                    
if __name__ == "__main__":
    main()
