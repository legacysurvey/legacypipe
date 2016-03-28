import os
import argparse
import glob
from astropy.io import fits
import numpy as np

parser = argparse.ArgumentParser(description="test")
parser.add_argument("-Ncopy",type=int,action="store",help='# of images to copy from /project to /scratch')
parser.add_argument("-band",choices=['g','R'],action="store",help='which band')
parser.add_argument("-ptf_field",choices=['p120001', 'p002740', 'p002845'],action="store",help='(RA,DEC)=(148.93, 1.651), (149.142857, 1.125),(149.142857, 3.375)')
parser.add_argument("-outdir",action="store",help='directory to make pimage/ and mask/ in for storing the copied images')
args = parser.parse_args()

for mydir in ['images']:
    if not os.path.exists(os.path.join(args.outdir,mydir)): os.makedirs(os.path.join(args.outdir,mydir))

key=dict(g='_f01_',R='_f02_')
ccd=dict(p120001='05',p002740='04',p002845='10') #each field happens to only have 1 ccd associated with it
copied=0
ptf_dir='/global/project/projectdirs/desi/imaging/data/ptf/cosmos/seeing_lt_3_airmass_lt_2'
fns= glob.glob(os.path.join(ptf_dir, 'PTF_*_scie_*%s*%s*_c%s.fits' %(key[args.band],args.ptf_field,ccd[args.ptf_field]) ))
if len(fns) == 0: print 'WARNING, 0 files found'
for fn in fns:
    data=fits.open(fn)
    if type(data[0].header['IMAGEZPT']) == np.float:    
        copied +=1
        cmd= ' '.join(['cp',fn,os.path.join(args.outdir,'images/',os.path.basename(fn))])
        if os.system(cmd): raise ValueError
        maskfn= fn.replace('_scie_','_mask_')
        cmd= ' '.join(['cp',maskfn,os.path.join(args.outdir,'images/',os.path.basename(maskfn))])
        if os.system(cmd): raise ValueError
    if copied >= args.Ncopy: break
if copied < args.Ncopy: print 'WARNING, only %d files found, you asked for %d' % (copied,args.Ncopy)
print "done"
