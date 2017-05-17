from __future__ import print_function
import sys
import os
import numpy as np
from glob import glob

from astropy.io import fits
# sha1sums
# nohup bash -c 'for num in `cat coadd_match_ep.txt|awk '"'"'{print $1}'"'"'`; do echo $num;find /global/cscratch1/sd/desiproc/dr4/data_release/dr4_fixes/coadd/$num -type f -print0|xargs -0 sha1sum > coadd_${num}_scr.sha1;done' > sha1.out &

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


def survey_bricks_bitmask():
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

def makedir_for_fn(fn): 
    try:
        os.makedirs(os.path.dirname(fn))
    except OSError:
        print('no worries, dir already exists %s' % os.path.dirname(fn))



def get_fns(brick,outdir):
    bri= brick[:3]
    trac= [os.path.join(outdir,'tractor-i',bri,'tractor-%s.fits' % brick) ]
    #os.path.join(outdir,'tractor',bri,'tractor-%s.fits' % brick),
    met= [os.path.join(outdir,'metrics',bri,'all-models-%s.fits' % brick),
          os.path.join(outdir,'metrics',bri,'blobs-%s.fits.gz' % brick)]
    coadd= glob(os.path.join(outdir,'coadd',bri,brick,'legacysurvey-%s-*.fits*' % brick))
    return trac + met + coadd
    #return trac
 
def get_new_fns(brick,outdir):
    bri= brick[:3]
    trac= [os.path.join(outdir,'tractor-i',bri,'tractor-%s.fits' % brick),
           os.path.join(outdir,'tractor',bri,'tractor-%s.fits' % brick)]
    met= [os.path.join(outdir,'metrics',bri,'all-models-%s.fits' % brick),
          os.path.join(outdir,'metrics',bri,'blobs-%s.fits.gz' % brick)]
    coadd= glob(os.path.join(outdir,'coadd',bri,brick,'legacysurvey-%s-*.fits*' % brick))
    return trac + met + coadd
    #return trac 

def get_touch_fn(brick,outdir):
    bri= brick[:3]
    fn= os.path.join(outdir,'touch_files',bri,'done-%s.txt' % brick)
    makedir_for_fn(fn)
    return fn
    
 
def get_sha_fn(brick,outdir):
    bri= brick[:3]
    return os.path.join(outdir,'tractor',bri, 'brick-%s.sha1sum' % brick)

def new_header(orig_fn, new_fn):
    # Copy dr4b file -> dr4c 
    # tractor.fits already there
    if orig_fn != new_fn:
        makedir_for_fn(new_fn)
        bash('cp %s %s' % (orig_fn, new_fn))
    
    # Gunzip
    is_gzip= 'fits.gz' in new_fn
    if is_gzip:
        bash('gunzip %s' % new_fn)
        new_fn= new_fn.replace('.gz','')

    # Header
    print('Editing %s' % new_fn)
    #a=fitsio.FITS(new_fn,'rw')
    #hdr=a[0].read_header()
    hdulist = fits.open(new_fn, mode='readonly') 
    # Skip if already fixed
    if 'RELEASE' in hdulist[0].header:
        pass
    elif 'DECALSDR' in hdulist[0].header:
        # Add
        for key,val,comm in zip(['RELEASE','SURVEYDR','SURVEYDT','SURVEY'],
                                ['4000','DR4',hdulist[0].header['DECALSDT'],'BASS MzLS'],
                                ['DR number','DR name','runbrick.py run time','Survey name']):
            hdulist[0].header.set(key,val,comm) 
        # Git describes: for bounding dates/commits when dr4b ran
        hdulist[0].header.set('ASTROMVR','0.67-188-gfcdd3c0, 0.67-152-gfa03658',
                              'astrometry_net (3/6-4/15/2017)')
        hdulist[0].header.set('TRACTOVR','dr4.1-9-gc73f1ab, dr4.1-9-ga5cfaa3', 
                              'tractor (2/22-3/31/2017)')
        hdulist[0].header.set('LEGDIRVR','dr3-17-g645c3ab', 
                              'legacypipe-dir (3/28/2017)')
        hdulist[0].header.set('LEGPIPVR','dr3e-834-g419c0ff, dr3e-887-g068df7a',
                              'legacypipe (3/15-4/19/2017)')
        # Remove
        rem_keys= ['DECALSDR','DECALSDT',
                   'SURVEYV','SURVEYID','DRVERSIO']
        for key in rem_keys:
            del hdulist[0].header[key]
        # Write
        #clob= False
        #if '/tractor/' in new_fn:
        #    clob=True
        clob= True
        hdulist.writeto(new_fn, clobber=clob, output_verify='fix')
        print('Wrote %s' % new_fn)

    # Gzip
    if is_gzip:
        bash('gzip %s' % new_fn) 


def tractor_i_fn(dir,brick):
    bri=brick[:3]
    return os.path.join(dir,'tractor-i',bri,'tractor-%s.fits' % brick)

def tractor_fn(dir,brick):
    bri=brick[:3]
    return os.path.join(dir,'tractor',bri,'tractor-%s.fits' % brick)

def do_checks(brick,dr4c_dir):
    # Sanity checks on outputs
    dr4c_fns= get_new_fns(brick,dr4c_dir)
    #size each file > 0
    #number files dr4c = dr4b
    #size dr4c ~ dr4b
    #number bricks that have nans, which cols nans appear in

def main(args=None):
    '''new data model catalouge and fix all headers
    touch a junk_brickname.txt file when done so can see which bricks finished
    OR 
    sanity checks on outputs
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bricklist', action='store',
                        help='text file listin bricknames to rewrite headers for')
    parser.add_argument('--sanitycheck', action='store_true',default=False,
                        help='set to test integrity of dr4c files')
    opt = parser.parse_args(args=args)

    dr4b_dir= '/global/projecta/projectdirs/cosmo/work/dr4b'
    dr4c_dir= '/global/projecta/projectdirs/cosmo/work/dr4c'

    bricks= np.loadtxt(opt.bricklist,dtype=str)
    assert(bricks.size > 0)
    if bricks.size == 1:
        bricks= np.array([bricks])
    for brick in bricks:
        if opt.sanitycheck:
            do_checks(brick,dr4c_dir=dr4c_dir)
            continue

        # New Data Model
        in_fn= tractor_i_fn(dr4b_dir,brick)
        out_fn= tractor_fn(dr4c_dir,brick)
        try:
            os.makedirs(os.path.dirname(out_fn))
        except OSError:
            print('no worries, dir already exists %s' % os.path.dirname(out_fn))
        bash('python legacypipe/format_catalog.py --in %s --out %s --dr4' % (in_fn,out_fn))
        
        # Fix Headers
        new_header(orig_fn=out_fn, new_fn=out_fn) #orig same as new fn
        # Everything else
        fns= get_fns(brick,outdir=dr4b_dir)
        for fn in fns:
            new_header(orig_fn=fn, new_fn=fn.replace('dr4b','dr4c')) #orig diff from new fn

        # Sha1sum 
        fns= get_new_fns(brick=brick, outdir=dr4c_dir)
        sha_fn= get_sha_fn(brick=brick, outdir=dr4c_dir)
        lis= ' '.join(fns)
        bash('echo %s |xargs sha1sum > %s' % (lis,sha_fn))
        #bash("find /global/cscratch1/sd/desiproc/dr4/data_release/dr4_fixes/coadd/$num -type f -print0|xargs -0 sha1sum > coadd_${num}_scr.sha1")
        print('Wrote sha_fn=%s' % sha_fn)

        # Touch a file so know that finished
        touch_fn= get_touch_fn(brick=brick, outdir=dr4c_dir)
        bash('touch %s' % touch_fn)
        print('Wrote %s' % touch_fn)
    
if __name__ == '__main__':
    main()
 
