


basedir='/project/projectdirs/uLens/ZTF/Tractor/data/ZTF18aajytjt_PS1stack/tractor/images/'
fns=glob.glob(basedir+'*stk*.new_sciimg.fits')
fns.append(glob.glob(basedir+'*stk*.weight.fits')[0])
print(fns)
with open('rings.v3.skycell.2381.062.wrp.g.56395_39459_sciimg.fits','rb') as f: 
	hdul=fits.open(f) 
	hdr=hdul[1].header 
	data=hdul[1].data 
	w.wcs.crpix = [hdr['CRPIX1'],hdr['CRPIX2']] 
	w.wcs.cdelt = np.array([hdr['CDELT1'],hdr['CDELT2']]) 
	w.wcs.crval = [hdr['CRVAL1'],hdr['CRVAL2']] 
	w.wcs.ctype = [hdr['CTYPE1'],hdr['CTYPE2']] 
	header = w.to_header() 
	print(header['CRPIX1'],header['CRPIX2']) 
	CD1_1=hdr['PC001001']*hdr['CDELT1'] 
	CD1_2=hdr['PC001002']*hdr['CDELT1'] 
	CD2_1=hdr['PC002001']*hdr['CDELT2'] 
	CD2_2=hdr['PC002002']*hdr['CDELT2']          
    hdu = fits.PrimaryHDU(data, header=hdr)
    


