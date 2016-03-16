import os
import argparse
import glob
from astropy.io import fits
import numpy as np

parser = argparse.ArgumentParser(description="test")
parser.add_argument("-mos_cp_dir",action="store",help='root dir of processed mosaic data')
parser.add_argument("-region",choices=['cosmos'],action="store",help='root dir of processed mosaic data')
args = parser.parse_args()

if args.region == 'cosmos': 
    c_ra,c_dec,dd= 150.2,2.2,0.5
    ra,dec= [c_ra-dd,c_ra+dd],[c_dec-dd,c_dec+dd]
print 'looking for images with ra,dec in between ra [%.1f,%.1f], dec [%.1f,%.1f]' % (ra[0],ra[1],dec[0],dec[1])

fns= glob.glob(os.path.join(args.mos_cp_dir, '*/','*ooi*.fits.fz'))
if len(fns) == 0: print 'WARNING, 0 files found'
print 'fns[0]= ',fns[0],'fns[-1]= ',fns[-1]
keep=[]
for cnt,fn in enumerate(fns):
    print 'reading image %d/%d, %s' % (cnt,len(fns),fn)
    data=fits.open(fn)
    img_ra,img_dec= data[0].header['CENTRA'],data[0].header['CENTDEC']
    if img_ra >= ra[0] and img_ra <= ra[1] and img_dec >= dec[0] and img_dec <= dec[1]: keep.append( fn ) 
print '%d/%d images in regions' % (len(keep),len(fns))
if len(keep) > 0:
    fout=open('images_in_%s.txt'% (args.region,),'w')
    for line in keep: 
        fout.write(line+'\n')
        fout.write(line.replace('ooi','oki')+'\n')
        fout.write(line.replace('ooi','ood')+'\n')
        fout.write(line.replace('ooi','oow')+'\n')
    fout.close()
print "done"
