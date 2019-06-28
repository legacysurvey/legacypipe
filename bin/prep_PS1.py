import numpy as np
from astropy.io import fits
dir = '/project/projectdirs/uLens/ZTF/Tractor/data/ZTF18aajytjt_PS1stack/tractor/images/'
fns=[dir+'rings.v3.skycell.2381.062.stk.g.unconv.sciimg.fits']
#fns=[dir+'rings.v3.skycell.2381.062.wrp.g.55676_42336_sciimg.fits',dir+'rings.v3.skycell.2381.062.wrp.g.55676_43545_sciimg.fits']
for fn in fns:
	with open(fn,'rb') as f: 
		hdul=fits.open(f)
		hdu = fits.PrimaryHDU(hdul[1].data)           
		hdu.header=hdul[1].header	
		hdu.header['EXTNAME']='CCD0'
		
		BZERO   =   3.454127907753E+00
		BSCALE  =   2.112995762387E-04
		BSOFTEN =   9.541472414983E+01
		BOFFSET =   2.246295928955E+00

		v = hdul[1].data #BZERO + BSCALE * hdul[1].data
		a = 1.0857362
		x = v/a
		flux = BOFFSET + BSOFTEN * 2 * np.sinh(x)                                
		flux = np.nan_to_num(flux)
		hdu.data=flux	
		
		print(hdul[1].header)                                  
		hdu.writeto(fn.rstrip('.sciimg.fits')+'.new_sciimg.fits',overwrite=True)      

fns=[dir+'rings.v3.skycell.2381.062.stk.g.unconv.mskimg.fits']
#fns=[dir+'rings.v3.skycell.2381.062.wrp.g.55676_42336_sciimg.fits',dir+'rings.v3.skycell.2381.062.wrp.g.55676_43545_sciimg.fits']
for fn in fns:
	with open(fn,'rb') as f: 
		hdul=fits.open(f)
		hdu = fits.PrimaryHDU(hdul[1].data, header=hdul[1].header)           
		#hdu.header=hdul[1].header	
		hdu.header['EXTNAME']='CCD0'
		hdu.writeto(fn.rstrip('.mskimg.fits')+'.new_mskimg.fits',overwrite=True)       

fns=[dir+'rings.v3.skycell.2381.062.stk.g.unconv.wt.fits']
#fns=[dir+'rings.v3.skycell.2381.062.wrp.g.55676_42336_sciimg.fits',dir+'rings.v3.skycell.2381.062.wrp.g.55676_43545_sciimg.fits']
for fn in fns:
	with open(fn,'rb') as f: 
		hdul=fits.open(f)
		print(hdul[0].header)
		hdu = fits.PrimaryHDU(hdul[1].data, header=hdul[1].header)           
		
		BZERO   =   3.454127907753E+00
		BSCALE  =   2.112995762387E-04
		BSOFTEN =   9.541472414983E+01
		BOFFSET =   2.246295928955E+00

		v = hdul[1].data #BZERO + BSCALE * hdul[1].data
		a = 1.0857362
		x = v/a
		flux = BOFFSET + BSOFTEN * 2 * np.sinh(x)                                
		#flux = np.nan_to_num(flux)
		hdu.data=flux	
         
		
		#hdu.header=hdul[1].header	
		hdu.header['EXTNAME']='CCD0'
		hdu.writeto(fn.rstrip('.wt.fits')+'.weight.fits',overwrite=True)       
 
