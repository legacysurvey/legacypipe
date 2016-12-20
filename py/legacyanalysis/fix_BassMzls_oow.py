from __future__ import division, print_function
import argparse
from astropy.io import fits
import numpy as np
import os

def dobash(cmd):
    print('UNIX cmd: %s' % cmd)
    if os.system(cmd): raise ValueError


parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.')
parser.add_argument('--dowhat',choices=['create','sanitycheck'],action='store',help='',required=True)
parser.add_argument('--imgfn',action='store',\
                    default='/project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP20160204v2/k4m_160205_083826_ooi_zd_v2.fits.fz',required=False)
args = parser.parse_args()

# Copy to SCRATCH so don't mess up original file
imgfn= args.imgfn
outdir=imgfn.replace('/project/projectdirs/cosmo/staging/',\
                     '/scratch2/scratchdirs/kaylanb/updatedwt/')
outdir=os.path.dirname(outdir)
if not os.path.exists(outdir):
    os.makedirs(outdir)
proj_dir= os.path.dirname(imgfn)
imgfn= os.path.basename(imgfn)
wtfn=imgfn.replace('ooi','oow')
dqfn=imgfn.replace('ooi','ood')
for fn in [imgfn,wtfn,dqfn]:
    if not os.path.exists(os.path.join(outdir,fn)):
        dobash("cp %s %s" % \
              (os.path.join(proj_dir,fn),\
               outdir))
imgfn= os.path.join(outdir,imgfn)
wtfn= os.path.join(outdir,wtfn)
dqfn= os.path.join(outdir,dqfn)
newfn= wtfn.replace('.fits.fz','_updatedwt.fits')
if args.dowhat == 'create':
    imgobj= fits.open(imgfn)
    wtobj = fits.open(wtfn)
    dqobj = fits.open(dqfn)
    hdr = imgobj[0].header
    #read_noise= hdr['RDNOISE'] # e
    #gain=1.8 # fixed
    const_sky= hdr['SKYADU'] # e/s, Recommended sky level keyword from Frank
    expt= hdr['EXPTIME'] # s
    for hdu in range(1,len(imgobj)):
        cpwt= wtobj[hdu].data # s/e, 1 / [var(sky) + var(read)]
        cpimg= imgobj[hdu].data # e/s, img - median bkgrd  - var(sky) - var(read) + const sky
        cpbad= dqobj[hdu].data # bitmask, 0 is good
        var_SR= 1./cpwt # e/s
        var_Astro= np.abs(cpimg - const_sky) / expt # e/s
        wt= 1./(var_SR + var_Astro) # s/e
        # Zero out NaNs and masked pixels
        wt[np.isfinite(wt) == False]= 0.
        wt[cpbad != 0]= 0.
        # Overwrite w new weight
        wtobj[hdu].data= wt
    wtobj.writeto(newfn)
    print('Wrote %s' % newfn)
elif args.dowhat == 'sanitycheck':
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    wtobj = fits.open(wtfn)
    newwtobj = fits.open(newfn)
    for hdu in range(1,len(wtobj)):
        cp_var= 1./wtobj[hdu].data
        new_var= 1./newwtobj[hdu].data
        assert( np.all(cp_var.flatten() >= 0.) ) 
        assert( np.all(new_var.flatten() >= 0.) )
        r=new_var/cp_var
        r= ma.masked_array(r, mask=np.isfinite(r) == False) 
        sz= float(r.shape[0]*r.shape[1])
        print('Fraction NaN=',np.where(r.mask == True)[0].size/sz)
        print('Fraction Finite=',np.where(r.mask == False)[0].size/sz)
        print('min,mean,max=',r.min(),r.mean(),r.max())
        fn= os.path.join(newfn.replace('.fits','_hist_hdu%d.png' % hdu))
        print('Making histrogram')
        bins= np.linspace(0.999,1.2,num=21)
        junk=plt.hist(r[100:200,100:200],bins=bins,normed=True,cumulative=True)
        plt.ylabel('Cumulative PDF')
        plt.xlabel('Var(sky,read,astro) / (1/oow)')
        plt.savefig(fn)
        print('Wrote %s' % fn)
    print('done')
        
        
    
    

#def read_fits(self, fn, hdu, slice=None, header=None, **kwargs):
#    if slice is not None:
#        f = fitsio.FITS(fn)[hdu]
#        img = f[slice]
#        rtn = img
#        if header:
#            hdr = f.read_header()
#            return (img,hdr)
#        return img
#    return fitsio.read(fn, ext=hdu, header=header, **kwargs)
