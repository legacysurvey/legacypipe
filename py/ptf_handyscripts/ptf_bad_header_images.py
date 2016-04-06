import os
import argparse
import glob
from astropy.io import fits
import numpy as np

parser = argparse.ArgumentParser(description="test")
parser.add_argument("-ptf_dir",action="store",help='dir with ptf images so $PTF')
args = parser.parse_args()

for mydir in ['bad_header_images']:
    if not os.path.exists(os.path.join(args.ptf_dir,mydir)): os.makedirs(os.path.join(args.ptf_dir,mydir))

fns= glob.glob(os.path.join(args.ptf_dir, '*_scie_*.fits'))
if len(fns) == 0: print 'WARNING, 0 files found'
f_copy=[]
for fn in fns:
    data=fits.open(fn)
    if type(data[0].header['IMAGEZPT']) != np.float:   
        f_copy.append( fn ) 
        cmd= ' '.join(['mv',fn,os.path.join(args.ptf_dir,'bad_header_images/')])
        if os.system(cmd): raise ValueError
        maskfn= fn.replace('_scie_','_mask_')
        cmd= ' '.join(['mv',maskfn,os.path.join(args.ptf_dir,'bad_header_images/')])
        if os.system(cmd): raise ValueError
print 'found %d bad header images, copied them to %s' % (len(f_copy),os.path.join(args.ptf_dir,'bad_header_images/'))
print "done"
