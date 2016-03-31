import os
import argparse
import glob
from astropy.io import fits
import numpy as np

def radec_to_float(ra,dec):
    '''ra = "10:30:00.00'" in hms
    dec = "22:30:23" in deg'''
    ra= np.array(ra.split(':')).astype(float)
    dec= np.array(dec.split(':')).astype(float)
    return 15*(ra[0]+ra[1]/60.+ra[2]/3600.), dec[0]+dec[1]/60.+dec[2]/3600. 


parser = argparse.ArgumentParser(description="test")
parser.add_argument("-search_str",action="store",help='root dir + search string for processed mosaic data')
parser.add_argument("-use_str_as",choices=['wildcard','filelist'],action="store",help='wildcard -- glob.glob(), filelist -- text file has fn line by line')
parser.add_argument("-region",choices=['cosmos','deep2_f3'],action="store",help='root dir of processed mosaic data')
args = parser.parse_args()

if args.region == 'cosmos': 
    c_ra,c_dec,dd= 150.2,2.2,0.5
    ra,dec= [c_ra-dd,c_ra+dd],[c_dec-dd,c_dec+dd]
    rakey,deckey= 'CENTRA','CENTDEC'
if args.region == 'deep2_f3': 
    c_ra,c_dec,dd= 352.,0.,2.
    ra,dec= [c_ra-dd,c_ra+dd],[c_dec-dd,c_dec+dd]
    rakey,deckey= 'RA','DEC'
print 'looking for images with ra,dec in between ra [%.1f,%.1f], dec [%.1f,%.1f]' % (ra[0],ra[1],dec[0],dec[1])

if args.use_str_as == 'wildcard': fns= glob.glob(args.search_str)
else: 
    fin=open(args.search_str,'r')
    fns= fin.readlines()
    fin.close()
    for i in range(len(fns)): fns[i]= fns[i].strip()
if len(fns) == 0: print 'WARNING, 0 files found'
print 'fns[0]= ',fns[0],'fns[-1]= ',fns[-1]
keep=[]
for cnt,fn in enumerate(fns):
    print 'reading image %d/%d, %s' % (cnt,len(fns),fn)
    data=fits.open(fn)
    try: 
        img_ra,img_dec= data[0].header[rakey],data[0].header[deckey]
        if args.region == 'deep2_f3': img_ra,img_dec= radec_to_float(img_ra,img_dec)
        #print 'img_ra= ',img_ra,'img_dec',img_dec
        if img_ra >= ra[0] and img_ra <= ra[1] and img_dec >= dec[0] and img_dec <= dec[1]: keep.append( fn ) 
        print 'max delta RA= ',max(abs(img_ra-ra[0]),abs(img_ra-ra[1])),'max delta DEC= ',max(abs(img_dec-dec[0]),abs(img_dec-dec[1])) 
    except KeyError: 
        print 'keys: %s,%s not found in file %s' % (rakey,deckey,fn)
        continue
print '%d/%d images in regions' % (len(keep),len(fns))
for k in keep: print k
print 'exiting early'
sys.exit(0)
if len(keep) > 0:
    fout=open('images_in_%s.txt'% (args.region,),'w')
    for line in keep: 
        fout.write(line+'\n')
        fout.write(line.replace('ooi','oki')+'\n')
        fout.write(line.replace('ooi','ood')+'\n')
        fout.write(line.replace('ooi','oow')+'\n')
    fout.close()
print "done"
