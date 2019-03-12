#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

from legacypipe.internal import sharedmem
from legacypipe.internal.io import iter_tractor, parse_filename

import argparse
import os, sys
from time import time

import fitsio

def main():
    ns = parse_args()
            
    if ns.ignore_errors:
        print("Warning: *** Will ignore broken tractor catalogue files ***")
        print("         *** Disable -I for final data product.         ***")
    # avoid each subprocess importing h5py again and again.
    if 'hdf5' in ns.format: 
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
        'blocks' : sweep_schema_blocks(36, 36),
        'dec' : sweep_schema_dec(180),
        }

    sweeps = schemas[ns.schema]

    t0 = time()

    nbricks_tot = np.zeros((), 'i8')
    nobj_tot = np.zeros((), 'i8')

    def work(sweep):
        data, header, nbricks = make_sweep(sweep, bricks, ns)

        header.update({
            'RAMIN'  : sweep[0],
            'DECMIN' : sweep[1],
            'RAMAX'  : sweep[2],
            'DECMAX' : sweep[3],
            })

        template = "sweep-%(ramin)s%(decmin)s-%(ramax)s%(decmax)s.%(format)s"

        def formatdec(dec):
            return ("%+04g" % dec).replace('-', 'm').replace('+', 'p')
        def formatra(ra):
            return ("%03g" % ra)

        for format in ns.format:
            filename = template %  \
                dict(ramin=formatra(sweep[0]), 
                     decmin=formatdec(sweep[1]),
                     ramax=formatra(sweep[2]), 
                     decmax=formatdec(sweep[3]),
                     format=format)

            if len(data) > 0:
                save_sweep_file(os.path.join(ns.dest, filename), 
                    data, header, format)

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
        bricksdesc = dict([(item['BRICKNAME'].decode(), item) for item in bricksdesc])
    else:
        bricksdesc = None
             
    #- Load list of bricknames to use
    if ns.bricklist is not None:
        bricklist = np.loadtxt(ns.bricklist, dtype='S8')
        # TODO: skip unknown bricks?
        d = dict([(brickname.decode(), d[brickname]) 
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

class NA: pass 

def make_sweep(sweep, bricks, ns):
    data = [np.empty(0, dtype=SWEEP_DTYPE)]
    header = {}
    ra1, dec1, ra2, dec2 = sweep
    def merge_header(header, header2):
        for key, value in header2.items():
            if key not in header:
                header[key] = value
            else:
                if header[key] is NA:
                    pass
                else:
                    if header[key] != value:
                        header[key] = NA
        
    with sharedmem.MapReduce(np=0) as pool:
        def filter(brickname, filename, region):
            if not intersect(sweep, region): 
                return None, None
            try:
                objects = fitsio.read(filename, 1, upper=True)
                chunkheader = fitsio.read_header(filename, 0, upper=True)
            except:
                if ns.ignore_errors:
                    print('IO error on %s' % filename)
                    return None, None
                else:
                    raise
            # ADM check all the column dtypes match.
            if not ns.ignore_errors:
                sflds = SWEEP_DTYPE.fields
                tflds = objects.dtype.fields
                for fld in sflds:
                    try:
                        sdt, tdt = sflds[fld][0], tflds[fld][0]
                        assert sdt == tdt
                    except:
                        msg = 'sweeps/Tractor dtypes differ for field '
                        msg += '{}. Sweeps: {}, Tractor: {}'.format(fld, sdt, tdt)
                        raise ValueError(msg)

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
            chunkheader = dict([(key, chunkheader[key]) for key in chunkheader.keys()])    
            return chunk, chunkheader

        def reduce(chunk, chunkheader):
            if chunk is not None:
                data.append(chunk)
                merge_header(header, chunkheader)
        pool.map(filter, bricks, star=True, reduce=reduce)

    neff = len(data) - 1

    data = np.concatenate(data, axis=0)
    header = dict([(key, value) for key, value in header.items() if value is not NA])
    return data, header, neff


def save_sweep_file(filename, data, header, format):
    if format == 'fits':
        header = [dict(name=key, value=header[key]) for key in sorted(header.keys())]
        with fitsio.FITS(filename, mode='rw', clobber=True) as ff:
            ff.create_image_hdu()
            ff[0].write_keys(header)
            ff.write_table(data, extname='SWEEP', header=header)

    elif format == 'hdf5':
        import h5py
        with h5py.File(filename, 'w') as ff:
            dset = ff.create_dataset('SWEEP', data=data)
            for key in header:
                dset.attrs[key] = header[key]
    else:
        raise ValueError("Unknown format")
    
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
#   ('BRICK_PRIMARY', '?'),
    ('RELEASE', '>i2'),
    ('BRICKID', '>i4'),
    ('BRICKNAME', 'S8'),
    ('OBJID', '>i4'),
    ('TYPE', 'S4'),
    ('RA', '>f8'),
    ('DEC', '>f8'),
    ('RA_IVAR', '>f4'),
    ('DEC_IVAR', '>f4'),
    ('DCHISQ', '>f4', (5,)),
    ('EBV', '>f4'),
#   ('TYCHO2INBLOB', '?'),
#   ('OUT_OF_BOUNDS', '?'),
#   ('DECAM_FLUX', '>f4', (6,)),
#   ('WISE_FLUX', '>f4', (4,)),
    ('FLUX_U', '>f4'),
    ('FLUX_G', '>f4'),
    ('FLUX_R', '>f4'),
    ('FLUX_I', '>f4'),
    ('FLUX_Z', '>f4'),
    ('FLUX_Y', '>f4'),
    ('FLUX_W1', '>f4'),
    ('FLUX_W2', '>f4'),
    ('FLUX_W3', '>f4'),
    ('FLUX_W4', '>f4'),
#   ('DECAM_FLUX_IVAR', '>f4', (6,)),
#   ('WISE_FLUX_IVAR', '>f4', (4,)),
    ('FLUX_IVAR_U', '>f4'),
    ('FLUX_IVAR_G', '>f4'),
    ('FLUX_IVAR_R', '>f4'),
    ('FLUX_IVAR_I', '>f4'),
    ('FLUX_IVAR_Z', '>f4'),
    ('FLUX_IVAR_Y', '>f4'),
    ('FLUX_IVAR_W1', '>f8'),
    ('FLUX_IVAR_W2', '>f8'),
    ('FLUX_IVAR_W3', '>f8'),
    ('FLUX_IVAR_W4', '>f8'),
#   ('DECAM_MW_TRANSMISSION', '>f4', (6,)),
#   ('WISE_MW_TRANSMISSION', '>f4', (4,)),
    ('MW_TRANSMISSION_U', '>f4'),
    ('MW_TRANSMISSION_G', '>f4'),
    ('MW_TRANSMISSION_R', '>f4'),
    ('MW_TRANSMISSION_I', '>f4'),
    ('MW_TRANSMISSION_Z', '>f4'),
    ('MW_TRANSMISSION_Y', '>f4'),
    ('MW_TRANSMISSION_W1', '>f4'),
    ('MW_TRANSMISSION_W2', '>f4'),
    ('MW_TRANSMISSION_W3', '>f4'),
    ('MW_TRANSMISSION_W4', '>f4'),
#   ('DECAM_NOBS', 'u1', (6,)),
#   ('WISE_NOBS', '>i2', (4,)), 
    ('NOBS_U', '>i2'),
    ('NOBS_G', '>i2'),
    ('NOBS_R', '>i2'),
    ('NOBS_I', '>i2'),
    ('NOBS_Z', '>i2'),
    ('NOBS_Y', '>i2'),
    ('NOBS_W1', '>i2'),
    ('NOBS_W2', '>i2'),
    ('NOBS_W3', '>i2'),
    ('NOBS_W4', '>i2'),
#   ('DECAM_RCHI2', '>f4', (6,)),
#   ('WISE_RCHI2', '>f4', (4,)),
    ('RCHISQ_U', '>f4'),
    ('RCHISQ_G', '>f4'),
    ('RCHISQ_R', '>f4'),
    ('RCHISQ_I', '>f4'),
    ('RCHISQ_Z', '>f4'),
    ('RCHISQ_Y', '>f4'),
    ('RCHISQ_W1', '>f4'),
    ('RCHISQ_W2', '>f4'),
    ('RCHISQ_W3', '>f4'),
    ('RCHISQ_W4', '>f4'),
#   ('DECAM_FRACFLUX', '>f4', (6,)),
#   ('WISE_FRACFLUX', '>f4', (4,)),
    ('FRACFLUX_U', '>f4'),
    ('FRACFLUX_G', '>f4'),
    ('FRACFLUX_R', '>f4'),
    ('FRACFLUX_I', '>f4'),
    ('FRACFLUX_Z', '>f4'),
    ('FRACFLUX_Y', '>f4'),
    ('FRACFLUX_W1', '>f4'),
    ('FRACFLUX_W2', '>f4'),
    ('FRACFLUX_W3', '>f4'),
    ('FRACFLUX_W4', '>f4'),
#   ('DECAM_FRACMASKED', '>f4', (6,)),
    ('FRACMASKED_U', '>f4'),
    ('FRACMASKED_G', '>f4'),
    ('FRACMASKED_R', '>f4'),
    ('FRACMASKED_I', '>f4'),
    ('FRACMASKED_Z', '>f4'),
    ('FRACMASKED_Y', '>f4'),
#   ('DECAM_FRACIN', '>f4', (6,)),
    ('FRACIN_U', '>f4'),
    ('FRACIN_G', '>f4'),
    ('FRACIN_R', '>f4'),
    ('FRACIN_I', '>f4'),
    ('FRACIN_Z', '>f4'),
    ('FRACIN_Y', '>f4'),
#   ('DECAM_ANYMASK', '>i2', (6,)),
    ('ANYMASK_U', '>i2'),
    ('ANYMASK_G', '>i2'),
    ('ANYMASK_R', '>i2'),
    ('ANYMASK_I', '>i2'),
    ('ANYMASK_Z', '>i2'),
    ('ANYMASK_Y', '>i2'),
#   ('DECAM_ALLMASK', '>i2', (6,)),
    ('ALLMASK_U', '>i2'),
    ('ALLMASK_G', '>i2'),
    ('ALLMASK_R', '>i2'),
    ('ALLMASK_I', '>i2'),
    ('ALLMASK_Z', '>i2'),
    ('ALLMASK_Y', '>i2'),
    ('WISEMASK_W1', 'u1'),
    ('WISEMASK_W2', 'u1'),
#   ('DECAM_PSFSIZE', '>f4', (6,)),
    ('PSFSIZE_U', '>f4'),
    ('PSFSIZE_G', '>f4'),
    ('PSFSIZE_R', '>f4'),
    ('PSFSIZE_I', '>f4'),
    ('PSFSIZE_Z', '>f4'),
    ('PSFSIZE_Y', '>f4'),
#   ('DECAM_DEPTH', '>f4', (6,)),
    ('PSFDEPTH_U', '>f4'),
    ('PSFDEPTH_G', '>f4'),
    ('PSFDEPTH_R', '>f4'),
    ('PSFDEPTH_I', '>f4'),
    ('PSFDEPTH_Z', '>f4'),
    ('PSFDEPTH_Y', '>f4'),
#   ('DECAM_GALDEPTH', '>f4', (6,)),
    ('GALDEPTH_U', '>f4'),
    ('GALDEPTH_G', '>f4'),
    ('GALDEPTH_R', '>f4'),
    ('GALDEPTH_I', '>f4'),
    ('GALDEPTH_Z', '>f4'),
    ('GALDEPTH_Y', '>f4'),
    ('WISE_COADD_ID', 'S8'),
    ('FRACDEV', '>f4'),
    ('FRACDEV_IVAR', '>f4'),
    ('SHAPEDEV_R', '>f4'),
    ('SHAPEDEV_R_IVAR', '>f4'),
    ('SHAPEDEV_E1', '>f4'),
    ('SHAPEDEV_E1_IVAR', '>f4'),
    ('SHAPEDEV_E2', '>f4'),
    ('SHAPEDEV_E2_IVAR', '>f4'),
    ('SHAPEEXP_R', '>f4'),
    ('SHAPEEXP_R_IVAR', '>f4'),
    ('SHAPEEXP_E1', '>f4'),
    ('SHAPEEXP_E1_IVAR', '>f4'),
    ('SHAPEEXP_E2', '>f4'),
    ('SHAPEEXP_E2_IVAR', '>f4'),
    ('FIBERFLUX_U', '>f4'),
    ('FIBERFLUX_G', '>f4'),
    ('FIBERFLUX_R', '>f4'),
    ('FIBERFLUX_I', '>f4'),
    ('FIBERFLUX_Z', '>f4'),
    ('FIBERFLUX_Y', '>f4'),
    ('FIBERTOTFLUX_U', '>f4'),
    ('FIBERTOTFLUX_G', '>f4'),
    ('FIBERTOTFLUX_R', '>f4'),
    ('FIBERTOTFLUX_I', '>f4'),
    ('FIBERTOTFLUX_Z', '>f4'),
    ('FIBERTOTFLUX_Y', '>f4'),
    ('REF_ID', '>i8'),
    ('GAIA_PHOT_G_MEAN_MAG', '>f4'),
    ('GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', '>f4'),
    ('GAIA_PHOT_BP_MEAN_MAG', '>f4'),
    ('GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', '>f4'),
    ('GAIA_PHOT_RP_MEAN_MAG', '>f4'),
    ('GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', '>f4'),
    ('GAIA_ASTROMETRIC_EXCESS_NOISE', '>f4'),
    ('GAIA_DUPLICATED_SOURCE', '>b1'),
    ('PARALLAX', '>f4'),
    ('PARALLAX_IVAR', '>f4'),
    ('PMRA', '>f4'),
    ('PMRA_IVAR', '>f4'),
    ('PMDEC', '>f4'),
    ('PMDEC_IVAR', '>f4'),
    ('BRIGHTBLOB', '>i2')]
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

    ap.add_argument('-f', "--format", choices=['fits', 'hdf5'], nargs='+', default=["fits"],
        help="Format of the output sweep files")

    ap.add_argument('-F', "--filelist", default=None,
        help="list of tractor brickfiles to use; this will avoid expensive walking of the path.")

    ap.add_argument('-d', "--bricksdesc", default=None, 
        help="location of decals-bricks.fits, speeds up the scanning")

    ap.add_argument('-v', "--verbose", action='store_true')
    ap.add_argument('-I', "--ignore-errors", action='store_true')

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
