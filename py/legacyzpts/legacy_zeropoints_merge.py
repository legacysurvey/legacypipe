#!/usr/bin/env python

"""
mpi gather all ccd fits files together and write single ccd table
"""
from __future__ import division, print_function

import os
import argparse
import numpy as np
from collections import defaultdict
try:
    from astrometry.util.fits import fits_table, merge_tables
except ImportError:
    pass

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    if len(lines) < 1: raise ValueError('lines not read properly from %s' % fn)
    return np.array( list(np.char.strip(lines)) )


def write_cat(cat, outname):
    outname= outname.replace('.fits','')
    fn= outname+'.fits'
    if os.path.exists(fn):
        os.remove(fn)
    cat.writeto(fn)
    print('Wrote %s' % fn)
    #os.system('gzip --best ' + fn)
    #print('Converted to %s.gz' % fn)

def fix_hdu_b4merge(zpt_one_exposure):
    import fitsio
    proj= '/project/projectdirs/cosmo/staging'
    proja= '/global/projecta/projectdirs/cosmo/staging'
    fn= zpt_one_exposure.image_filename[0].strip()
    if os.path.exists( os.path.join(proj,fn) ):
        fn= os.path.join(proj,fn)
    elif os.path.exists( os.path.join(proja,fn) ):
        fn= os.path.join(proja,fn)
    else:
        raise ValueError('could not find this fn on proj or proja! %s' % fn)
    hdulist= fitsio.FITS(fn)
    for i,ccdname in enumerate(zpt_one_exposure.ccdname):
        zpt_one_exposure.image_hdu[i]= hdulist[ccdname.strip()].get_extnum()

def fix_hdu_post(tab):
    import fitsio
    proj= '/project/projectdirs/cosmo/staging'
    proja= '/global/projecta/projectdirs/cosmo/staging'
    hdus=[]
    #tab= tab[:120]
    for cnt,T in enumerate(tab):
        if cnt % 100 == 0:
            print('%d/%d' % (cnt,len(tab)))
        fn= T.image_filename.strip() 
        ccdname= T.ccdname.strip().upper()
        try: 
            hdulist= fitsio.FITS(os.path.join(proj,fn))
        except IOError:
            try: 
                hdulist= fitsio.FITS(os.path.join(proja,fn))
            except:
                raise IOError('could not find this fn on proj or proja! %s' % fn)
        hdus.append( hdulist[ccdname].get_extnum() )
    tab.set('image_hdu', np.array(hdus))
    return tab

def main(args):
    parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.')
    parser.add_argument('--file_list',help='List of zeropoint fits files to concatenate')
    parser.add_argument('--nproc',type=int,default=1,help='number mpi tasks')
    parser.add_argument('--outname', type=str, default='combined_legacy_zpt.fits', help='Output directory.')
    parser.add_argument('--fix_hdu', action='store_true',default=False, help='')
    parser.add_argument('--remove-file-prefix', help='Remove prefix from image_filename, if present')
    parser.add_argument('--cut', action='store_true',default=False, help='Cut to ccd_cuts==0')
    parser.add_argument('--cut-expnum', action='store_true',default=False, help='Cut out rows with expnum==0')
    parser.add_argument('files', nargs='*', help='Zeropoint files to concatenate')

    opt = parser.parse_args(args)

    fns = np.array(opt.files)
    if opt.file_list:
        fns = np.hstack([fns, read_lines(opt.file_list)])

    # RUN HERE
    if opt.nproc > 1:
        from mpi4py.MPI import COMM_WORLD as comm
        print('rank %d' % comm.rank)
        fn_split= np.array_split(fns, comm.size)
        cats=[]
        for cnt,fn in enumerate( fn_split[comm.rank] ):
            if comm.rank == 0:
                #if cnt % 100 == 0:
                print('%d/%d' % (cnt,len(fn_split[comm.rank])))
            try:
                if opt.fix_hdu:
                    T= fits_table(fn)
                    fix_hdu_b4merge(T)
                    cats.append( T )
                else:
                    cats.append( fits_table(fn) )
            except IOError:
                print('File is bad, skipping: %s' % fn)
        cats= merge_tables(cats, columns='fillzero')
        #if opt.fix_hdu:
        #    print('fixing hdu')
        #    cats= fix_hdu(cats)
        # Gather
        all_cats = comm.gather( cats, root=0 )
        if comm.rank == 0:
            all_cats= merge_tables(all_cats, columns='fillzero')
            write_cat(all_cats,outname=opt.outname)
            print("Done")
    else:
        cats=[]
        for cnt,fn in enumerate(fns):
            if (cnt % 500) == 0:
                print('Reading CCDs table %d/%d' % (cnt,len(fns)))
            t = fits_table(fn)
            cats.append(t)
        cats= merge_tables(cats, columns='fillzero')
        if opt.fix_hdu:
            print('fixing hdu')
            cats= fix_hdu(cats)
        if opt.cut:
            print(len(cats), 'CCDs')
            cats.cut(cats.ccd_cuts == 0)
            print(len(cats), 'CCDs pass ccd_cuts == 0')
        if opt.cut_expnum:
            print(len(cats), 'CCDs')
            cats.cut(cats.expnum > 0)
            print(len(cats), 'CCDs have expnum > 0')
        if opt.remove_file_prefix:
            fns = []
            for fn in cats.image_filename:
                if fn.startswith(opt.remove_file_prefix):
                    fn = fn.replace(opt.remove_file_prefix, '', 1)
                fns.append(fn)
            cats.image_filename = np.array(fns)
        write_cat(cats, outname=opt.outname)
        print("Done")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
