from __future__ import print_function
import sys
import os
import numpy as np
import fitsio
from glob import glob


def survey_ccd_bitmask():
    mzls=fitsio.FITS('survey-ccds-mzls2.fits','rw')
    hdr= mzls[1].read_header()
    if not 'BITMASK' in hdr:
        mzls[1].write_key('PHOTOME','Photometric',comment='True if CCD considered photometric')
        mzls[1].write_key('BITMASK','Cuts besides Photometric',comment='See below')
        mzls[1].write_key('BIT1','bad_expid.txt file')
        mzls[1].write_key('BIT2','ccd_hdu_mismatch')
        mzls[1].write_key('BIT3','zpts_bad_astrom')
    mzls.close

def survey_ccd_add_bitmask():
    mzls=fits_table("survey-ccds-mzls2.fits")
    bm= np.zeros(len(mzls)).astype(np.uint8)
    bm[ mzls.bad_expid ]+= 1
    bm[ mzls.ccd_hdu_mismatch ]+= 2
    bm[ mzls.zpts_bad_astrom ]+= 4
    mzls.set('bitmask', bm)
    for key in ['bad_expid','ccd_hdu_mismatch','zpts_bad_astrom']:
        mzls.delete(key)


def survey_bricks_bitmask
    b=fitsio.FITS('survey-bricks-dr4.fits.gz','rw')
    hdr= b[1].read_header()
    if not 'BITMASK' in hdr:
        b[1].write_key('BITMASK','All flagged CCDs thrown out in DR4',comment='See below')
        b[1].write_key('BIT1','dr4',comment='have tractor catalogue')
        b[1].write_key('BIT2','oom',comment='failed b/c out of memory')
        b[1].write_key('BIT3','old_chkpt',comment='failed b/c old checkpoint when reran')
        b[1].write_key('BIT4','no_signif_srcs',comment='failed b/c all detected sources not significant')
        b[1].write_key('BIT5','no_srcs',comment='failed b/c no sources detected')
    b.close

def survey_brick_add_bitmask():
    bricks=fits_table("survey-bricks-dr4.fits.gz")
    bm= np.zeros(len(bricks)).astype(np.uint8)
    for key,bitval in zip(['cut_dr4','cut_oom','cut_old_chkpt',
                           'cut_no_signif_srcs','cut_no_srcs'],
                          [1,2,4,
                           8,16]):
        bm[ bricks.get(key) ]+= bitval
        bricks.delete(key)
    bricks.set('bitmask', bm)


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
 
