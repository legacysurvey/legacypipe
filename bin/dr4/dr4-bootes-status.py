from __future__ import print_function
import argparse
import os
import numpy as np
from astrometry.util.fits import fits_table, merge_tables

parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.') 
parser.add_argument('--camera', type=str, default='90prime') 
parser.add_argument('--get_bricks_notdone', action='store_true', default=False) 
args = parser.parse_args() 

if args.get_bricks_notdone:
    b=fits_table(os.path.join(os.environ['LEGACY_SURVEY_DIR'],'survey-bricks-%s.fits.gz' % args.camera))
    don=np.loadtxt('bricks_done_%s.tmp' % args.camera,dtype=str)
    fout= 'bricks_notdone_%s.tmp' % args.camera
    if os.path.exists(fout):
        os.remove(fout)
    # Bricks not finished
    with open(fout,'w') as fil:
        for brick in list( set(b.brickname).difference( set(don) ) ):
            fil.write('%s\n' % brick)
    print('Wrote %s' % fout)
    # All Bricks
    fout= 'bricks_all_%s.tmp' % args.camera
    if os.path.exists(fout):
        exit()
    with open(fout,'w') as fil:
        for brick in b.brickname:
            fil.write('%s\n' % brick)
    print('Wrote %s' % fout)
    
    


     
