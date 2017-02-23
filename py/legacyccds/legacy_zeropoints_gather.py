#!/usr/bin/env python

"""
mpi gather all ccd fits files together and write single ccd table
"""
from __future__ import division, print_function

import os
import argparse
import numpy as np
from astrometry.util.fits import fits_table, merge_tables

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    if len(lines) < 1: raise ValueError('lines not read properly from %s' % fn)
    return np.array( list(np.char.strip(lines)) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.')
    parser.add_argument('--file_list',action='store',help='List of zeropoint fits files to concatenate',required=True)
    parser.add_argument('--nproc',type=int,action='store',default=1,help='number mpi tasks',required=True)
    parser.add_argument('--outname', type=str, default='combined_legacy_zpt.fits', help='Output directory.')
    opt = parser.parse_args()
    
    fns= read_lines(opt.file_list) 

    # RUN HERE
    if opt.nproc > 1:
        from mpi4py.MPI import COMM_WORLD as comm
        fn_split= np.array_split(fns, comm.size)
        cats=[]
        for fn in fn_split[comm.rank]:
            try:
                cats.append( fits_table(fn) )
            except IOError:
                print('File is bad, skipping: %s' % fn)
        cats= merge_tables(cats, columns='fillzero')
        # Gather
        all_cats = comm.gather( cats, root=0 )
        if comm.rank == 0:
            all_cats= merge_tables(all_cats, columns='fillzero')
            if os.path.exists(opt.outname):
                os.remove(opt.outname)
            all_cats.writeto(opt.outname)
            print('Wrote %s' % opt.outname)
            print("Done")
    else:
        cats=[]
        for cnt,fn in enumerate(fns):
            print('Reading %d/%d' % (cnt,len(fns)))
            cats.append( fits_table(fn) )
        cats= merge_tables(cats, columns='fillzero')
        if os.path.exists(opt.outname):
            os.remove(opt.outname)
        cats.writeto(opt.outname)

