#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

from legacyproduct.internal import sharedmem

import argparse
import os, sys
from time import time

import fitsio

def main():
    ns = parse_args()
            
    # avoid each subprocess importing h5py again and again.
    if ns.format == 'h5py': 
        import h5py

    # this may take a while on a file system with slow meta-data 
    # access
    # bricks = [(name, filepath, region), ...]
    bricks = list_bricks(ns)

    t0 = time()
            
    try:
        os.makedirs(ns.dest)
    except OSError:
        pass

    # blocks or ra stripes?
    schemas = {
        'ra' : sweep_schema_ra(360),
        'blocks' : sweep_schema_blocks(36, 10),
        'dec' : sweep_schema_dec(180),
        }

    sweeps = schemas[ns.schema]

    t0 = time()

    nbricks_tot = np.zeros((), 'i8')
    nobj_tot = np.zeros((), 'i8')

    def work(sweep):
        filename = ns.template %  \
            dict(ramin=sweep[0], decmin=sweep[1],
                 ramax=sweep[2], decmax=sweep[3],
                 format=ns.format)


        data, nbricks = make_sweep(sweep, bricks, ns)

        header = {
            'RAMIN'  : sweep[0],
            'DECMIN' : sweep[1],
            'RAMAX'  : sweep[2],
            'DECMAX' : sweep[3],
            }

        if len(data) > 0:
            save_sweep_file(os.path.join(ns.dest, filename), 
                data, header, ns.format)
        return filename, nbricks, len(data)

    def reduce(filename, nbricks, nobj): 
        nbricks_tot[...] += nbricks
        nobj_tot[...] += nobj

        if ns.verbose and nobj > 0:
            print (
            '%s : %d bricks %d primary objects, %g bricks / sec %g objs / sec' % 
            ( filename, nbricks, nobj, 
              nbricks_tot / (time() - t0), 
              nobj_tot / (time() - t0), 
            )
            )

    with sharedmem.MapReduce(np=ns.numproc) as pool:
        pool.map(work, sweeps, reduce=reduce)


def list_bricks(ns):
    t0 = time()

    if ns.filelist is not None:
        d = dict([(parse_filename(fn.strip()), fn.strip()) 
            for fn in open(ns.filelist, 'r').readlines()])
    else:
        d = dict(iter_tractor(ns.src))

    if ns.verbose:
        print('enumerated %d bricks in %g seconds' % (
            len(d), time() - t0))

    if ns.bricksdesc is not None:
        bricksdesc = fitsio.read(ns.bricksdesc, 1, upper=True)
        bricksdesc = dict([(item['BRICKNAME'], item) for item in bricksdesc])
    else:
        bricksdesc = None
             
    #- Load list of bricknames to use
    if ns.bricklist is not None:
        bricklist = np.loadtxt(ns.bricklist, dtype='S8')
        # TODO: skip unknown bricks?
        d = dict([(brickname, d[brickname]) 
                             for brickname in bricklist])

    t0 = time()

    with sharedmem.MapReduce(np=ns.numproc) as pool:
        chunksize = 1024
        keys = list(d.keys())
        def work(i):
            return [
                (brickname, d[brickname], read_region(brickname, d[brickname], bricksdesc))
                for brickname in keys[i:i+chunksize]
                ]
        bricks = sum(pool.map(work, range(0, len(keys), chunksize)), [])

    if ns.verbose:
        print('read regions of %d bricks in %g seconds' % (
            len(bricks), time() - t0))
 
    return bricks

def sweep_schema_ra(nstripes):
    ra = np.linspace(0, 360, nstripes + 1, endpoint=True)
    return [(ra[i], -90, ra[i+1], 90) for i in range(len(ra) - 1)]

def sweep_schema_dec(nstripes):
    dec = np.linspace(-90, 90, nstripes + 1, endpoint=True)
    return [(0, dec[i], 360, dec[i+1]) for i in range(len(dec) - 1)]

def sweep_schema_blocks(nra, ndec):
    ra = np.linspace(0, 360, nra + 1, endpoint=True)
    dec = np.linspace(-90, 90, ndec + 1, endpoint=True)
    
    return [(ra[i], dec[j], ra[i+1], dec[j+1]) for i in range(len(ra) - 1) for j in range(len(dec) - 1)]

def make_sweep(sweep, bricks, ns):
    data = [np.empty(0, dtype=SWEEP_DTYPE)]
    ra1, dec1, ra2, dec2 = sweep

    with sharedmem.MapReduce(np=0) as pool:
        def filter(brickname, filename, region):
            if not intersect(sweep, region): 
                return None
            objects = fitsio.read(filename, 1, upper=True)

            mask = objects['BRICK_PRIMARY'] != 0
            objects = objects[mask]
            mask = objects['RA'] >= ra1
            mask &= objects['RA'] < ra2
            mask &= objects['DEC'] >= dec1
            mask &= objects['DEC'] < dec2
            objects = objects[mask]

            chunk = np.empty(len(objects), dtype=SWEEP_DTYPE)

            for colname in chunk.dtype.names:
                if colname not in objects.dtype.names:
                    # skip missing columns 
                    continue
                try:
                    chunk[colname][...] = objects[colname][...]
                except ValueError:
                    print('failed on column `%s`' % colname)
                    raise
                    
            return chunk
        def reduce(chunk):
            if chunk is not None:
                data.append(chunk)

        pool.map(filter, bricks, star=True, reduce=reduce)

    neff = len(data) - 1

    data = np.concatenate(data, axis=0)
    return data, neff


def save_sweep_file(filename, data, header, format):
    if format == 'fits':
        fitsio.write(filename, data, extname='SWEEP', header=header, clobber=True)
    elif format == 'hdf5':
        import h5py
        with h5py.File(filename, 'w') as ff:
            dset = ff.create_dataset('SWEEP', data=data)
            for key in header:
                dset.attrs[key] = header[key]
    else:
        raise ValueError("Unknown format")

import re
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
    
def intersect(region1, region2, tol=0.1):
    #  ra1, dec1, ra2, dec2 = region1
    # left top right bottom
    def pad(r):
        # this do not use periodic boundary yet
        return (r[0] - tol, r[1] - tol, r[2] + tol, r[3] + tol)

    region1 = pad(region1) 
    region2 = pad(region2) 

    dx = min(region1[2], region2[2]) - max(region1[0], region2[0])
    dy = min(region1[3], region2[3]) - max(region1[1], region2[1]) 
    return (dx > 0) & (dy > 0)

def read_region(brickname, filename, bricksdesc):
    if bricksdesc is not None:
        item = bricksdesc[brickname]
        return item['RA1'], item['DEC1'], item['RA2'], item['DEC2']

    f = fitsio.FITS(filename)
    h = f[0].read_header()
    r = h['RAMIN'], h['DECMIN'], h['RAMAX'], h['DECMAX']
    f.close()
    return r

SWEEP_DTYPE = np.dtype([
    ('BRICKID', '>i4'), 
    ('BRICKNAME', 'S8'), 
    ('OBJID', '>i4'), 
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
    ('DECAM_PSFSIZE', '>f4', (6,)), 
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
    ('DCHISQ', '>f4', (5,)), 
    ('FRACDEV', '>f4'), 
    ('TYCHO2INBLOB', '?'), 
    ('SHAPEDEV_R', '>f4'), 
    ('SHAPEEXP_R', '>f4'), 
    ('EBV', '>f4')]
)

def parse_args():
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

    ap.add_argument('-F', "--filelist", default=None,
        help="list of tractor brickfiles to use; this will avoid expensive walking of the path.")

    ap.add_argument('-t', "--template", 
        default="sweep%(ramin)+04g%(decmin)+03g%(ramax)+04g%(decmax)+03g.%(format)s",
        help="Tempalte of the output file name")

    ap.add_argument('-d', "--bricksdesc", default=None, 
        help="location of decals-bricks.fits, speeds up the scanning")

    ap.add_argument('-v', "--verbose", action='store_true')

    ap.add_argument('-S', "--schema", choices=['blocks', 'dec', 'ra'], 
            default='blocks', 
            help="""Decomposition schema. Still being tuned. """)

    ap.add_argument('-b', "--bricklist", 
        help="""Filename with list of bricknames to include. 
                If not set, all bricks in src are included, sorted by brickname.
            """)

    ap.add_argument("--numproc", type=int, default=None,
        help="""Number of concurrent processes to use. 0 for sequential execution. 
            Default is to use OMP_NUM_THREADS, or the number of cores on the node.""")

    return ap.parse_args()

if __name__ == "__main__":
    main()
