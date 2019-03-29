#!/usr/bin/env python
"""
Write image file lists.

"""
import os, pdb
import argparse
import numpy as np
from astropy.table import Table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', choices=['decam', 'mosaic'],
                        action='store', required=True)
    args = parser.parse_args()

    if args.camera == 'decam':
        vsuffix = 'v4'
    else:
        vsuffix = 'v2'
    ondisklist = os.path.join('image-lists', 'dr8-ondisk-{}-{}.fits'.format(args.camera, vsuffix))
    data = Table.read(ondisklist)
    imfile = np.array([os.path.basename(ff).strip() for ff in data['filename']])

    ccds = Table.read('old/survey-ccds-dr8b-90prime-mosaic-nocuts.fits')
    ann = Table.read('old/annotated-ccds-dr8b-90prime-mosaic-nocuts.fits')
    ccdfile = [os.path.basename(ff).strip() for ff in ccds['image_filename']]

    drop, keep = [], []
    for ii, ccdfile1 in enumerate(ccdfile):
        this = np.where(np.isin(imfile, ccdfile1))[0]
        if data['qkeep'][this] == False:
            #print('Dropping {}'.format(ccdfile1))
            drop.append(ii)
        else:
            keep.append(ii)

        #if 'k4m_160602_050454_ooi_zd_v2' in ccdfile1:
        #    pdb.set_trace()

    drop = np.array(drop)
    keep = np.array(keep)
    print(len(drop), len(keep), len(ccds))

    ccds[keep].write('survey-ccds-dr8b-90prime-mosaic-nocuts.fits', overwrite=True)
    ann[keep].write('annotated-ccds-dr8b-90prime-mosaic-nocuts.fits', overwrite=True)
            
    pdb.set_trace()
        
if __name__ == '__main__':
    main()
