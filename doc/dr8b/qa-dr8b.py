#!/usr/bin/env python

"""
Visualize the brick and CCD coverage of the test regions.

"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import os, shutil, pdb
from glob import glob
import numpy as np
from astrometry.util.fits import fits_table
import fitsio
from astropy.table import Table, hstack
import matplotlib.pyplot as plt

def main():

    drdir = '/global/project/projectdirs/cosmo/work/legacysurvey/dr8b'

    missing_decam = [] # missing Tractor catalogs / partially completed bricks
    missing_mosaic = [] # missing Tractor catalogs / partially completed bricks

    done_decam = []
    done_mosaic = []
    
    fix_release_decam = []
    fix_release_mosaic = []

    release = {'decam': 8000, '90prime-mosaic': 8001}

    check_release = False # True
    remove_incomplete = False # True
    remove_pickles = True
    write_missing_bricks = False
    
    for camera in ('decam', '90prime-mosaic'):
        print(camera.upper())
        #for region in np.atleast_1d('dr8-test-overlap'):
        for region in ('dr8-test-overlap', 'dr8-test-s82', 'dr8-test-hsc-sgc', 'dr8-test-hsc-ngc',
                       'dr8-test-edr', 'dr8-test-hsc-north', 'dr8-test-deep2-egs'):
            
            # The 90prime+mosaic overlap brickfile has 427 bricks, not the 393 we ran with 
            if 'overlap' in region:
                brickfile = 'dr8b/bricks-{}-decam.fits'.format(region)
            else:
                brickfile = 'dr8b/bricks-{}-{}.fits'.format(region, camera)
                
            if os.path.isfile(brickfile):
                bricks = fits_table(brickfile)
                catfiles_all = np.array(['{}/runbrick-{}/tractor/{}/tractor-{}.fits'.format(drdir, camera, brick[:3], brick) for brick in bricks.brickname])
    
                catfiles_done = []
                for brick in bricks.brickname:
                    dd = os.path.join(drdir, '{}/runbrick-{}/tractor/???/tractor-{}.fits'.format(drdir, camera, brick))
                    # If the checksum exists, then the brick completed.
                    cc = dd.replace('tractor-', 'brick-').replace('.fits', '.sha256sum')
                    if len(glob(cc)):
                        ff = glob('{}'.format(dd))[0]
                        catfiles_done.append(ff)
                        if check_release:
                            rr = fitsio.read(ff, columns='release', lower=True, rows=0)
                            if rr[0] != release[camera]:
                                #print('Warning! {} has release={}'.format(os.path.basename(ff), rr[0]))
                                if camera == 'decam':
                                    fix_release_decam.append(brick)
                                else:
                                    fix_release_mosaic.append(brick)
                            
                catfiles_done = np.array(catfiles_done)
                
                ndone = len(catfiles_done)
                if ndone > 0:
                    print('  {:18}: {:04d}/{:04d} bricks done.'.format(region, ndone, len(bricks)))
                    if camera == 'decam':
                        done_decam.append(catfiles_done)
                    else:
                        done_mosaic.append(catfiles_done)
                        
                    miss = ~np.isin(catfiles_all, catfiles_done)
                    if np.sum(miss) > 0:
                        if camera == 'decam':
                            missing_decam.append(catfiles_all[miss])
                        else:
                            missing_mosaic.append(catfiles_all[miss])
        
    done_decam = np.hstack(done_decam)                
    done_mosaic = np.hstack(done_mosaic)
    missing_decam = np.hstack(missing_decam)                
    missing_mosaic = np.hstack(missing_mosaic)

    if write_missing_bricks:
        with open(os.path.join(drdir, 'brick-lists', 'bricks-dr8b-unfinished-decam'), 'w') as bb:
            for ff in missing_decam:
                brick = os.path.basename(ff).replace('tractor-', '').replace('.fits', '')
                bb.write('{}\n'.format(brick))
        
        with open(os.path.join(drdir, 'brick-lists', 'bricks-dr8b-unfinished-90prime-mosaic'), 'w') as bb:
            for ff in missing_mosaic:
                brick = os.path.basename(ff).replace('tractor-', '').replace('.fits', '')
                bb.write('{}\n'.format(brick))

    # Now do some clean-up.  Remove pickle files, checkpoints, and incomplete bricks.
    if remove_incomplete:
        for camera in ('decam', '90prime-mosaic'):
            print('Cleaning up incomplete bricks {}'.format(camera.upper()))
            if camera == 'decam':
                done = done_decam
                miss = missing_decam
            else:
                done = done_mosaic
                miss = missing_mosaic
    
            # If the brick didn't finish, remove the partial outputs, if any.
            for ff in miss:
                if os.path.isfile(ff):
                    print(ff)
                    raise ValueError()
                    
                brick = os.path.basename(ff).replace('tractor-', '').replace('.fits', '')
                coadd = os.path.join(os.path.dirname(ff).replace('tractor/', 'coadd/'), brick)
                if os.path.isdir(coadd):
                    print('  '+coadd)
                    #os.system('ls -l {}'.format(coadd))
                    #shutil.rmtree(coadd)
                tractori = os.path.join(os.path.dirname(ff).replace('tractor/', 'tractor-i'), brick)
                if os.path.isdir(tractori):
                    print('  '+tractori)
                    #shutil.rmtree(tractori)
    
    if remove_pickles:
        for camera in ('decam', '90prime-mosaic'):
            print('Cleaning up pickles {}'.format(camera.upper()))
            if camera == 'decam':
                done = done_decam
                miss = missing_decam
            else:
                done = done_mosaic
                miss = missing_mosaic

            # If the brick completed, clean it up!
            for ff in done:
                if not os.path.isfile(ff):
                    print(ff)
                    raise ValueError()
                
                brick = os.path.basename(ff).replace('tractor-', '').replace('.fits', '')
                pickles = glob(ff.replace('tractor/', 'pickles/').replace('tractor-', 'runbrick-').replace('.fits', '-*.pickle'))
                if len(pickles) > 0:
                    #print(pickles[0])
                    for pp in pickles:
                        os.remove(pp)
                        
                checkpt = glob(ff.replace('tractor/', 'checkpoints/').replace('tractor-', 'checkpoint-').replace('.fits', '.pickle'))
                if len(checkpt) > 0:
                    #print(checkpt)
                    os.remove(checkpt[0])
                        
    ## Stack the Tractor catalogs and make sure the data model is consistent.
    #for ff in done_mosaic:
    #    rr = fitsio.read(ff, columns='release', lower=True, rows=0)
    #    if rr[0] != 8001:
    #        print('Warning! {} has release={}'.format(os.path.basename(ff), rr[0]))
        
if __name__ == '__main__':
    main()
