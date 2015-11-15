#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

from legacyproduct.internal import sharedmem

import argparse
import os, sys
from time import time

import fitsio

SWEEP_DTYPE = np.dtype([
    ('BRICKID', '>i4'), 
    ('BRICKNAME', 'S8'), 
    ('OBJID', '>i4'), 
    ('BRICK_PRIMARY', '?'), 
    ('TYPE', 'S4'), 
    ('RA', '>f8'), 
    ('RA_IVAR', '>f4'), 
    ('DEC', '>f8'), 
    ('DEC_IVAR', '>f4'), 
    ('DECAM_FLUX', '>f4', (6,)), 
    ('DECAM_FLUX_IVAR', '>f4', (6,)), 
    ('DECAM_MW_TRANSMISSION', '>f4', (6,)), 
    ('DECAM_NOBS', 'u1', (6,)), 
    ('DECAM_RCHI2', '>f4', (6,)), 
    ('DECAM_FRACFLUX', '>f4', (6,)), 
    ('DECAM_FRACMASKED', '>f4', (6,)), 
    ('DECAM_FRACIN', '>f4', (6,)), 
    ('OUT_OF_BOUNDS', '?'), 
    ('DECAM_ANYMASK', '>i2', (6,)), 
    ('DECAM_ALLMASK', '>i2', (6,)), 
    ('WISE_FLUX', '>f4', (4,)), 
    ('WISE_FLUX_IVAR', '>f4', (4,)), 
    ('WISE_MW_TRANSMISSION', '>f4', (4,)), 
    ('WISE_NOBS', '>i2', (4,)), 
    ('WISE_FRACFLUX', '>f4', (4,)), 
    ('WISE_RCHI2', '>f4', (4,)), 
    ('DCHISQ', '>f4', (4,)), 
    ('FRACDEV', '>f4'), 
    ('EBV', '>f4')]
)

ap = argparse.ArgumentParser(
description="""Create Sweep files for DECALS. 
    This tool ensures each sweep file contains roughly '-n' objects. HDF5 and FITS formats are supported.
    Columns contained in a sweep file are: 

    [%(columns)s].
""" % dict(columns=str(SWEEP_DTYPE.names)),
    )

### ap.add_argument("--type", choices=["tractor"], default="tractor", help="Assume a type for src files")

ap.add_argument("src", help="Path to the root directory contains all tractor files")
ap.add_argument("dest", help="Path to the Output sweep file")

ap.add_argument('-f', "--format", choices=['fits', 'hdf5'], default="fits",
    help="Format of the output sweep files")

ap.add_argument('-u', "--unordered", action='store_true',
    help="Allow reordering of the input files; runs faster but may be undesirable.")

ap.add_argument('-n', "--nobjects-per-sweep", dest='nobjs', default=1024*512, type=int, 
    help="""Expected number of objects in a sweep file. 
            Will fail if -n is smaller than the number of objects in a tractor file.
        """)

ap.add_argument('-v', "--verbose", action='store_true')
ap.add_argument('-b', "--bricklist", 
    help="""Filename with list of bricknames to include. 
            If not set, all bricks in src are included, sorted by brickname.
        """)

ap.add_argument("--numproc", type=int, default=None,
    help="""Number of concurrent processes to use. 0 for sequential execution. 
        Default is to use OMP_NUM_THREADS, or the number of cores on the node.""")

def main():
    ns = ap.parse_args()
            
    # avoid each subprocess importing h5py again and again.
    if ns.format == 'h5py': 
        import h5py

    t0 = time()

    # this may take a while on a file system with slow meta-data 
    # access
    bricks = dict(iter_tractor(ns.src))

    print('enumerated %d bricks in %g seconds' % (
            len(bricks), time() - t0))

    #- Load list of bricknames to use
    if ns.bricklist is not None:
        bricklist = np.loadtxt(ns.bricklist, dtype='S8')
        # TODO: skip unknown bricks?
        list_of_bricks = [(brickname, bricks[brickname]) 
                             for brickname in bricklist]
    else:
        list_of_bricks = sorted(bricks.items())

    t0 = time()
            
    try:
        os.makedirs(ns.dest)
    except OSError:
        pass
    # (ns.nobjs - buffer_used) is the free buffer
    # we always round off at full bricks

    buffer = sharedmem.empty(ns.nobjs, dtype=SWEEP_DTYPE)
    buffer_used = sharedmem.empty((), dtype=np.intp)
    buffer_used[...] = 0

    # used in the filename of a new sweep file
    sweepfile_id = sharedmem.empty((), dtype=np.intp)
    sweepfile_id[...] = 0

    # number of brick files scanned, used for progress reporting.
    bricks_scanned = sharedmem.empty((), dtype=np.intp)
    bricks_scanned[...] = 0

    with sharedmem.MapReduce(np=ns.numproc) as pool:
        def filter(brickname, filename):
            objects = fitsio.read(filename, 1, upper=True)

            chunk = np.empty(len(objects), dtype=buffer.dtype)

            for colname in chunk.dtype.names:
                if colname not in objects.dtype.names:
                    # skip missing columns 
                    continue
                chunk[colname][...] = objects[colname][...]
                
                if len(chunk) > len(buffer):
                    raise RuntimeError("--number-of-objects per sweep file is too small")

            if not ns.unordered:
                protection = pool.ordered
            else:
                protection = pool.critical

            with protection:
                bricks_scanned[...] += 1
                if len(chunk) + buffer_used > len(buffer):
                    data = buffer[:buffer_used].copy()
                    sweep_filename = os.path.join(ns.dest, 
                        'sweep-%08d.%s' % (sweepfile_id, ns.format))
                    
                    buffer[:len(chunk)] = chunk
                    buffer_used[...] = len(chunk)
                    sweepfile_id[...] += 1
                else:
                    data = None
                    buffer[buffer_used:buffer_used + len(chunk)] = chunk
                    buffer_used[...] += len(chunk)

            # if data is set we shall write a sweep file
            if data is not None:
                header = {}
                save_sweep_file(sweep_filename, data, header, ns.format)

                rate = bricks_scanned / (time() - t0)
                if ns.verbose:
                    print ('%d objs saved in %s' % (len(data), sweep_filename))
                    print('%d bricks; %g bricks/sec' % (bricks_scanned, rate))
 
        pool.map(filter, list_of_bricks, star=True)

    if ns.verbose:
        print ('written to', ns.dest)

def save_sweep_file(filename, data, header, format):
    if format == 'fits':
        fitsio.write(filename, data, extname='SWEEP', header=header, clobber=True)
    elif format == 'hdf5':
        import h5py
        with h5py.File(filename, 'w') as ff:
            dset = ff.create_dataset('SWEEP', data=data)
            for key in header:
                dset[key] = header[key]
    else:
        raise ValueError("Unknown format")

import re
def iter_tractor(root):
    """ Iterator over all tractor files in a directory.

        Parameters
        ----------
        root : string
            Path to start looking
        
        Returns
        -------
        An iterator of (brickname, filename).

        Examples
        --------
        >>> for brickname, filename in iter_tractor('./'):
        >>>     print(brickname, filename)
        
        Notes
        -----
        root can be a directory or a single file; both create an iterator
    """
    def parse_filename(filename):
        """parse filename to check if this is a tractor brick file;
        returns brickname if it is, otherwise raises ValueError"""
        if not filename.endswith('.fits'): raise ValueError
        #- match filename tractor-0003p027.fits -> brickname 0003p027
        match = re.search('tractor-(\d{4}[pm]\d{3})\.fits', 
                os.path.basename(filename))

        if not match: raise ValueError

        brickname = match.group(1)
        return brickname

    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            for filename in filenames:
                try:
                    brickname = parse_filename(filename)
                    yield brickname, os.path.join(dirpath, filename)
                except ValueError:
                    #- not a brick file but that's ok; keep going
                    pass
    else:
        try:
            brickname = parse_filename(os.path.basename(root))
            yield brickname, root
        except ValueError:
            pass
    

if __name__ == "__main__":
    main()
