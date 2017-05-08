from __future__ import print_function
import sys
import os
import numpy as np
import fitsio
from glob import glob

def bash(cmd):
    print(cmd)
    rtn = os.system(cmd)
    if rtn:
        raise RuntimeError('Command failed: %s: return value: %i' %
                           (cmd,rtn))


def get_fns(brick,outdir):
    bri= brick[:3]
    lis= [os.path.join(outdir,'tractor-i',bri,'tractor-%s.fits' % brick),
          os.path.join(outdir,'tractor',bri,'tractor-%s.fits' % brick),
          os.path.join(outdir,'metrics',bri,'all-models-%s.fits' % brick),
          os.path.join(outdir,'metrics',bri,'blobs-%s.fits.gz' % brick)
         ]
    lis2= glob(os.path.join(outdir,'coadd',bri,brick,'legacysurvey-%s-*.fits*' % brick))
    return lis + lis2
    #return lis[:2]
    

def new_header(fn):
    is_gzip= 'fits.gz' in fn
    # Handle gzip
    if is_gzip:
        bash('gunzip %s' % fn)
        fn_gz= fn
        fn= fn.replace('.gz','')

    # Fix headers
    print('Editing %s' % fn)
    a=fitsio.FITS(fn,'rw')
    hdr=a[0].read_header()
    # Skip if already fixed
    if 'RELEASE' in hdr:
        pass
    elif 'DECALSDR' in hdr:
        # Add
        for key,val,comm in zip(['RELEASE','SURVEYDR','SURVEYDT','SURVEY'],
                                ['4000','DR4',hdr['DECALSDT'],'BASS MzLS'],
                                ['DR number','DR name','runbrick.py run time','Survey name']):
            a[0].write_key(key,val, comment=comm) 
        # Git describes: for bounding dates/commits when dr4b ran
        a[0].write_key('ASTROMVR','0.67-188-gfcdd3c0, 0.67-152-gfa03658',
                       comment='astrometry_net (3/6-4/15/2017)')
        a[0].write_key('TRACTOVR','dr4.1-9-gc73f1ab, dr4.1-9-ga5cfaa3',
                       comment='tractor (2/22-3/31/2017)')
        a[0].write_key('LEGDIRVR','dr3-17-g645c3ab',
                       comment= 'legacypipe-dir (3/28/2017)')
        a[0].write_key('LEGPIPVR','dr3e-834-g419c0ff, dr3e-887-g068df7a',
                       comment='legacypipe (3/15-4/19/2017)')
        # Remove
        rem_keys= ['DECALSDR','DECALSDT',
                   'SURVEYV','SURVEYID','DRVERSIO']
        for key in rem_keys:
            #a[0].delete(key)
            a[0].write_key(key,'', comment=' ') 
    # Close
    a.close()

    # Re-zip if need to
    if is_gzip:
        bash('gzip %s' % fn) 

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bricklist', action='store',
                        help='text file listin bricknames to rewrite headers for')
    parser.add_argument('--outdir', action='store',
                        help='abs path to data release files')
    opt = parser.parse_args(args=args)

    bricks= np.loadtxt(opt.bricklist,dtype=str)
    assert(bricks.size > 0)
    if bricks.size == 1:
        bricks= np.array([bricks])
    for brick in bricks:
        fns= get_fns(brick,outdir=opt.outdir)
        for fn in fns:
            new_header(fn)

if __name__ == '__main__':
    main()
 
