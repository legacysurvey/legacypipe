from math import *
import numpy as np
import healpy as hp
import astropy.io.fits as pyfits
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import numpy.random
import os, errno
import subprocess
twopi = 2.*pi
piover2 = .5*pi
verbose = False


# ---------------------------------------------------------------------------------------- #
def quicksipVerbose(verb=False):
    global verbose
    verbose=verb

# Make directory
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

# Some unit definitions
arcsec_to_radians = 0.0000048481368111
degree_to_arcsec = 3600.0

# MarcM Global variable to debug
#nwrong = 0

# ---------------------------------------------------------------------------------------- #

# Write partial Healpix map to file
# indices are the indices of the pixels to be written
# values are the values to be written
def write_partial_map(filename, indices, values, nside, nest=False):
    fitsformats = [hp.fitsfunc.getformat(np.int32), hp.fitsfunc.getformat(np.float32)]
    column_names = ['PIXEL', 'SIGNAL']
    # maps must have same length
    assert len(set((len(indices), len(values)))) == 1, "Indices and values must have same length"
    if nside < 0:
        raise ValueError('Invalid healpix map : wrong number of pixel')
    firstpix = np.min(indices)
    lastpix = np.max(indices)
    npix = np.size(indices)
    cols=[]
    for cn, mm, fm in zip(column_names, [indices, values], fitsformats):
        cols.append(pyfits.Column(name=cn, format='%s' % fm, array=mm))
    if False: # Deprecated : old way to create table with pyfits before v3.3
        tbhdu = pyfits.new_table(cols)
    else:
        tbhdu = pyfits.BinTableHDU.from_columns(cols)
    # add needed keywords
    tbhdu.header['PIXTYPE'] = ('HEALPIX','HEALPIX pixelisation')
    if nest: ordering = 'NESTED'
    else:    ordering = 'RING'
    tbhdu.header['ORDERING'] = (ordering, 'Pixel ordering scheme, either RING or NESTED')
    tbhdu.header['EXTNAME'] = ('xtension', 'name of this binary table extension')
    tbhdu.header['NSIDE'] = (nside,'Resolution parameter of HEALPIX')
    tbhdu.header['FIRSTPIX'] = (firstpix, 'First pixel # (0 based)')
    tbhdu.header['OBS_NPIX'] = npix
    tbhdu.header['GRAIN'] = 1
    tbhdu.header['OBJECT'] = 'PARTIAL'
    tbhdu.header['INDXSCHM'] = ('EXPLICIT', 'Indexing: IMPLICIT or EXPLICIT')
    tbhdu.writeto(filename, overwrite=True)
    subprocess.call("gzip -f "+filename,shell=True)

# ---------------------------------------------------------------------------------------- #

# Find healpix ring number from z
def ring_num(nside, z, shift=0):
    # ring = ring_num(nside, z [, shift=])
    #     returns the ring number in {1, 4*nside-1}
    #     from the z coordinate
    # usually returns the ring closest to the z provided
    # if shift = -1, returns the ring immediatly north (of smaller index) of z
    # if shift = 1, returns the ring immediatly south (of smaller index) of z

    my_shift = shift * 0.5
    # equatorial
    iring = np.round( nside*(2.0 - 1.5*z) + my_shift )
    if (z > 2./3.):
        iring = np.round( nside * np.sqrt(3.0*(1.0-z)) + my_shift )
        if (iring == 0):
            iring = 1
    # south cap
    if (z < -2./3.):
       iring = np.round( nside * np.sqrt(3.0*(1.0+z)) - my_shift )
       if (iring == 0):
           iring = 1
       iring = int(4*nside - iring)
    # return ring number
    return int(iring)

# ---------------------------------------------------------------------------------------- #

# returns the z coordinate of ring ir for Nside
def  ring2z (nside, ir):
    fn = float(nside)
    if (ir < nside): # north cap
       tmp = float(ir)
       z = 1.0 - (tmp * tmp) / (3.0 * fn * fn)
    elif (ir < 3*nside): # tropical band
       z = float( 2*nside-ir ) * 2.0 / (3.0 * fn)
    else:   # polar cap (south)
       tmp = float(4*nside - ir )
       z = - 1.0 + (tmp * tmp) / (3.0 * fn * fn)
    # return z
    return z

# ---------------------------------------------------------------------------------------- #

def ang2pix_ring_ir(nside,ir,phi):
#    c=======================================================================
#   c     gives the pixel number ipix (RING) 
#    c     corresponding to angles theta and phi
#    c=======================================================================
	z = ring2z (nside, ir)
	z0=2.0/3.0
	za = fabs(z)
	if phi >= twopi:
		phi = phi - twopi
	if phi < 0.:
		phi = phi + twopi
	tt = phi / piover2#;//  ! in [0,4)
  
	nl2 = 2*nside
	nl4 = 4*nside
	ncap  = nl2*(nside-1)#// ! number of pixels in the north polar cap
	npix  = 12*nside*nside
  
	if za <= z0:# {
		jp = int(floor(nside*(0.5 + tt - z*0.75)))#; /*index of ascending edge line*/
		jm = int(floor(nside*(0.5 + tt + z*0.75)))#; /*index of descending edge line*/
    
		#ir = nside + 1 + jp - jm#;// ! in {1,2n+1} (ring number counted from z=2/3)
		kshift = 0
		if fmod(ir,2)==0.:
			kshift = 1#;// ! kshift=1 if ir even, 0 otherwise
		ip = int(floor( ( jp+jm - nside + kshift + 1 ) / 2 ) + 1)#;// ! in {1,4n}
		if ip>nl4:
			ip = ip - nl4
    
		ipix1 = ncap + nl4*(ir-1) + ip

	else:
    
		tp = tt - floor(tt)#;//      !MOD(tt,1.d0)
		tmp = sqrt( 3.*(1. - za) )
		
		jp = int(floor( nside * tp * tmp ))#;// ! increasing edge line index
		jm = int(floor( nside * (1. - tp) * tmp ))#;// ! decreasing edge line index
		
		#ir = jp + jm + 1#;//        ! ring number counted from the closest pole
		ip = int(floor( tt * ir ) + 1)#;// ! in {1,4*ir}
		if ip>4*ir:
			ip = ip - 4*ir
		
		ipix1 = 2*ir*(ir-1) + ip
		if z<=0.:
			ipix1 = npix - 2*ir*(ir+1) + ip

	return ipix1 - 1


# gives the list of Healpix pixels contained in [phi_low, phi_hi]
def in_ring_simp(nside, iz, phi_low, phi_hi, conservative=True):
	pixmin = int(ang2pix_ring_ir(nside,iz,phi_low))
	pixmax = int(ang2pix_ring_ir(nside,iz,phi_hi))
	if pixmax < pixmin:
		pixmin1 = pixmax
		pixmax = pixmin
		pixmin = pixmin1
	listir = np.arange(pixmin, pixmax)
	return listir
	
# gives the list of Healpix pixels contained in [phi_low, phi_hi]
def in_ring(nside, iz, phi_low, phi_hi, conservative=True):
# nir is the number of pixels found
# if no pixel is found, on exit nir =0 and result = -1
    if phi_hi-phi_low == 0:
    	return -1
    npix = hp.nside2npix(nside)
    ncap  = 2*nside*(nside-1) # number of pixels in the north polar cap
    listir = -1
    nir = 0

    # identifies ring number
    if ((iz >= nside) and (iz <= 3*nside)): # equatorial region
        ir = iz - nside + 1  # in {1, 2*nside + 1}
        ipix1 = ncap + 4*nside*(ir-1) #  lowest pixel number in the ring
        ipix2 = ipix1 + 4*nside - 1   # highest pixel number in the ring
        kshift = ir % 2
        nr = nside*4
    else:
        if (iz < nside): #  north pole
            ir = iz
            ipix1 = 2*ir*(ir-1)        #  lowest pixel number in the ring
            ipix2 = ipix1 + 4*ir - 1   # highest pixel number in the ring
        else:                         #    south pole
            ir = 4*nside - iz
            ipix1 = npix - 2*ir*(ir+1) #  lowest pixel number in the ring
            ipix2 = ipix1 + 4*ir - 1   # highest pixel number in the ring
        nr = int(ir*4)
        kshift = 1

    twopi = 2.*np.pi
    shift = kshift * .5
    if conservative:
        # conservative : include every intersected pixels,
        # even if pixel CENTER is not in the range [phi_low, phi_hi]
        ip_low = round (nr * phi_low / twopi - shift)
        ip_hi  = round (nr * phi_hi  / twopi - shift)
        ip_low = ip_low % nr      # in {0,nr-1}
        ip_hi  = ip_hi  % nr      # in {0,nr-1}
    else:
        # strict : include only pixels whose CENTER is in [phi_low, phi_hi]
        ip_low = np.ceil (nr * phi_low / twopi - shift)
        ip_hi  = np.floor(nr * phi_hi  / twopi - shift)
        diff = (ip_low - ip_hi) % nr      # in {-nr+1,nr-1}
        if (diff < 0):
            diff = diff + nr # in {0,nr-1}
        if (ip_low >= nr):
            ip_low = ip_low - nr
        if (ip_hi  < 0 ):
            ip_hi  = ip_hi  + nr
        #print ip_hi-ip_low,nr
    if phi_low <= 0.0 and phi_hi >= 2.0*np.pi:
        ip_low = 0
        ip_hi = nr - 1
    if (ip_low > ip_hi):
        to_top = True
    else:
        to_top = False
    ip_low = int( ip_low + ipix1 )
    ip_hi  = int( ip_hi  + ipix1 )

    ipix1 = int(ipix1)
    if (to_top):
        nir1 = int( ipix2 - ip_low + 1 )
        nir2 = int( ip_hi - ipix1  + 1 )
        nir  = int( nir1 + nir2 )
        if ((nir1 > 0) and (nir2 > 0)):
            listir   = np.concatenate( (np.arange(ipix1, nir2+ipix1), np.arange(ip_low, nir1+ip_low) ) )
        else:
            if nir1 == 0:
                listir   = np.arange(ipix1, nir2+ipix1)
            if nir2 == 0:
                listir   = np.arange(ip_low, nir1+ip_low)
    else:
        nir = int(ip_hi - ip_low + 1 )
        listir = np.arange(ip_low, nir+ip_low)
    #below added by AJR to address region around ra = 360
    if float(listir[-1]-listir[0])/(ipix2-ipix1) > .5:
    	listir1 = np.arange(ipix1, listir[0]+1)
    	listir2 = np.arange(listir[-1], ipix2+1)
    #	#print listir[-1],listir[0],ipix1,ipix2,len(listir1),len(listir2)
    	listir   = np.concatenate( (listir1,listir2  ) )   
    	#print len(listir)
    return listir

# ---------------------------------------------------------------------------------------- #

# Linear interpolation
def lininterp(xval, xA, yA, xB, yB):

    slope = (yB-yA) / (xB-xA)
    yval = yA + slope * (xval - xA)
    return yval

# ---------------------------------------------------------------------------------------- #

# Test if val beints to interval [b1, b2]
def inInter(val, b1, b2):
    if b1 <= b2:
        return np.logical_and( val <= b2, val >= b1 )
    else:
        return np.logical_and( val <= b1, val >= b2 )

# ---------------------------------------------------------------------------------------- #

# Test if a list of (theta,phi) values below to a region defined by its corners (theta,phi) for Left, Right, Bottom, Upper
def in_region(thetavals, phivals, thetaU, phiU, thetaR, phiR, thetaL, phiL, thetaB, phiB):

    npts = len(thetavals)
    phis = np.ndarray( (npts, 4) )
    thetas = np.ndarray( (npts, 4) )
    inds_phi = np.ndarray( (npts, 4), dtype=bool )
    inds_phi[:,:] = False
    inds_theta = np.ndarray( (npts, 4), dtype=bool )
    inds_theta[:,:] = False

    if thetaU != thetaB:
        phis[:,0] = lininterp(thetavals, thetaB, phiB, thetaU, phiU)
        inds_phi[:,0] = inInter(thetavals, thetaB, thetaU)
    if thetaL != thetaU:
        phis[:,1] = lininterp(thetavals, thetaU, phiU, thetaL, phiL)
        inds_phi[:,1] = inInter(thetavals, thetaU, thetaL)
        inds_phi[phis[:,0]==phis[:,1],1] = False
    if thetaL != thetaR:
        phis[:,2] = lininterp(thetavals, thetaL, phiL, thetaR, phiR)
        inds_phi[:,2] = inInter(thetavals, thetaL, thetaR)
        inds_phi[phis[:,0]==phis[:,2],2] = False
        inds_phi[phis[:,1]==phis[:,2],2] = False
    if thetaR != thetaB:
        phis[:,3] = lininterp(thetavals, thetaR, phiR, thetaB, phiB)
        inds_phi[:,3] = inInter(thetavals, thetaR, thetaB)
        inds_phi[phis[:,0]==phis[:,3],3] = False
        inds_phi[phis[:,1]==phis[:,3],3] = False
        inds_phi[phis[:,2]==phis[:,3],3] = False

    if phiU != phiB:
        thetas[:,0] = lininterp(phivals, phiB, thetaB, phiU, thetaU)
        inds_theta[:,0] = inInter(phivals, phiB, phiU)
    if phiL != phiU:
        thetas[:,1] = lininterp(phivals, phiU, thetaU, phiL, thetaL)
        inds_theta[:,1] = inInter(phivals, phiU, phiL)
        inds_theta[thetas[:,0]==thetas[:,1],1] = False
    if phiL != phiR:
        thetas[:,2] = lininterp(phivals, phiL, thetaL, phiR, thetaR)
        inds_theta[:,2] = inInter(phivals, phiL, phiR)
        inds_theta[thetas[:,0]==thetas[:,2],2] = False
        inds_theta[thetas[:,1]==thetas[:,2],2] = False
    if phiR != phiB:
        thetas[:,3] = lininterp(phivals, phiR, thetaR, phiB, thetaB)
        inds_theta[:,3] = inInter(phivals, phiR, phiB)
        inds_theta[thetas[:,0]==thetas[:,3],3] = False
        inds_theta[thetas[:,1]==thetas[:,3],3] = False
        inds_theta[thetas[:,2]==thetas[:,3],3] = False

    ind = np.where(np.logical_and(inds_phi[:,:].sum(axis=1)>1, inds_theta[:,:].sum(axis=1)>1))[0]
    res = np.ndarray( (npts, ), dtype=bool )
    res[:] = False

    for i in ind:
        phival = phivals[i]
        thetaval = thetavals[i]
        phis_loc = phis[i,inds_phi[i,:]]
        thetas_loc = thetas[i,inds_theta[i,:]]
        res[i] = (phival >= phis_loc[0]) & (phival <= phis_loc[1]) & (thetaval >= thetas_loc[0]) & (thetaval <= thetas_loc[1])

    return res

# ---------------------------------------------------------------------------------------- #

# Computes healpix pixels of propertyArray.
# pixoffset is the number of pixels to truncate on the edges of each ccd image.
# ratiores is the super-resolution factor, i.e. the edges of each ccd image are processed
#   at resultion 4*nside and then averaged at resolution nside.
#def computeHPXpix_sequ_new(nside, propertyArray, pixoffset=0, ratiores=4, coadd_cut=True):
def computeHPXpix_sequ_new(nside, propertyArray, pixoffset=0, ratiores=4, coadd_cut=False): 
    #return 'ERROR'
    #img_ras, img_decs = [propertyArray[v] for v in ['ra0', 'ra1', 'ra2','ra3']],[propertyArray[v] for v in ['dec0', 'dec1', 'dec2','dec3']]
    #x = [1+pixoffset, propertyArray['NAXIS1']-pixoffset, propertyArray['NAXIS1']-pixoffset, 1+pixoffset, 1+pixoffset]
    #y = [1+pixoffset, 1+pixoffset, propertyArray['NAXIS2']-pixoffset, propertyArray['NAXIS2']-pixoffset, 1+pixoffset]

    #if np.any(img_ras > 360.0):
    #    img_ras[img_ras > 360.0] -= 360.0
    #if np.any(img_ras < 0.0):
    #    img_ras[img_ras < 0.0] += 360.0
    #print 'in here'
    #print len(img_ras)#,len(img_ras[0])
    #plt.plot(img_ras[0],img_decs[0],'k,')
    #plt.show()
    img_ras, img_decs = computeCorners_WCS_TPV(propertyArray, pixoffset)
     
    #DEBUGGING - MARCM 
    #print "debugging img_ras img_decs", img_ras
    #for i in range(0,len(img_ras)):
    # 	if img_ras[i] > 360.:
    #		img_ras[i] -= 360.
    # 	if img_ras[i] < 0.:
    #		img_ras[i] += 360.
    #END DEBUGGING MARCM BIT 


    # Coordinates of coadd corners
    # RALL, t.DECLL, t.RAUL, t.DECUL, t.RAUR, t.DECUR, t.RALR, t.DECLR, t.URALL, t.UDECLL, t.URAUR, t.UDECUR
    if coadd_cut:
        #coadd_ras = [propertyArray[v] for v in ['URAUL', 'URALL', 'URALR', 'URAUR']]
        #coadd_decs = [propertyArray[v] for v in ['UDECUL', 'UDECLL', 'UDECLR', 'UDECUR']]
        coadd_ras = [propertyArray[v] for v in ['ra0', 'ra1', 'ra2', 'ra3']]
        coadd_decs = [propertyArray[v] for v in ['dec0', 'dec1', 'dec2', 'dec3']]
        coadd_phis = np.multiply(coadd_ras, np.pi/180)
        coadd_thetas =  np.pi/2  - np.multiply(coadd_decs, np.pi/180)
    else:
        coadd_phis = 0.0
        coadd_thetas = 0.0
    # Coordinates of image corners
    #print img_ras
    img_phis = np.multiply(img_ras , np.pi/180)
    img_thetas =  np.pi/2  - np.multiply(img_decs , np.pi/180)


    img_pix = hp.ang2pix(nside, img_thetas, img_phis, nest=False)
    pix_thetas, pix_phis = hp.pix2ang(nside, img_pix, nest=False)

    # DEBUGGING - MARCM
    #print 'pix_thetas', pix_thetas
    #print 'pix_phis', pix_phis
    #sys.exit()

    #img_phis = np.mod( img_phis + np.pi, 2*np.pi ) # Enable these two lines to rotate everything by 180 degrees
    #coadd_phis = np.mod( coadd_phis + np.pi, 2*np.pi ) # Enable these two lines to rotate everything by 180 degrees

    # MARCM patch to correct a bug from Boris which didn't get bass and mzls ccds corners properly oriented. 
    # This patch is not necesarily comprehensive; not pairing may not cover all cases
    # In addition it also needs checking what hapens around phi=0  
    dph01=abs(img_phis[0]-img_phis[1])
    dph12=abs(img_phis[1]-img_phis[2])

    if (dph01 < dph12) : 
        if (img_phis[1] < img_phis[2]): 
            if(img_thetas[0] < img_thetas[1]):
                # this was original bit
                #print "This is DECaLS" 
                ind_U = 0
                ind_L = 2
                ind_R = 3
                ind_B = 1
            else:
                # This is for MzLS (seems to rotate other way)  
                #print "This is MzLS"
                ind_U = 1 
                ind_L = 3 
                ind_R = 2 
                ind_B = 0
           #     print "Probably wrong indexing of ccd corner AAA" 
        else:
            # This is addes for BASS
            #print "This is for BASS" 
            if(img_thetas[0] > img_thetas[1]):
                ind_U = 2 
                ind_L = 0
                ind_R = 1
                ind_B = 3
            else:
                # Few o(100) ccd of DECaLS z-band fall here; not clear what to do on them 
                #ind_U = 3
                #ind_L = 1
                #ind_R = 0
                #ind_B = 2
                ind_U = 0
                ind_L = 2
                ind_R = 3
                ind_B = 1
    else:
        print("WARNING: (MARCM:) Current ccd image may have wrong corner assignments in quicksip")
        #raise ValueError("(MARCM:) probably wrong assignment of corner values in quicksip")
        #ind_U = 0
        #ind_L = 2
        #ind_R = 3
        #ind_B = 1
        ind_U = 3
        ind_L = 1
        ind_R = 0
        ind_B = 2

    ipix_list = np.zeros(0, dtype=int)
    weight_list = np.zeros(0, dtype=float)
    # loop over rings until reached bottom
    iring_U = ring_num(nside, np.cos(img_thetas.min()), shift=0)
    iring_B = ring_num(nside, np.cos(img_thetas.max()), shift=0)
    ipixs_ring = []
    pmax = np.max(img_phis)
    pmin = np.min(img_phis)
    if (pmax - pmin > np.pi):
        ipixs_ring = np.int64(np.concatenate([in_ring(nside, iring, pmax, pmin, conservative=True) for iring in range(iring_U-1, iring_B+1)]))
    else:
        ipixs_ring = np.int64(np.concatenate([in_ring(nside, iring, pmin, pmax, conservative=True) for iring in range(iring_U-1, iring_B+1)]))

    ipixs_nest = hp.ring2nest(nside, ipixs_ring)
    npixtot = hp.nside2npix(nside)
    if ratiores > 1:
        subipixs_nest = np.concatenate([np.arange(ipix*ratiores**2, ipix*ratiores**2+ratiores**2, dtype=np.int64) for ipix in ipixs_nest])
        nsubpixperpix = ratiores**2
    else:
        subipixs_nest = ipixs_nest
        nsubpixperpix = 1

    rangepix_thetas, rangepix_phis = hp.pix2ang(nside*ratiores, subipixs_nest, nest=True)
    #subipixs_ring = hp.ang2pix(nside*ratiores, rangepix_thetas, rangepix_phis, nest=False).reshape(-1, nsubpixperpix)

    if (pmax - pmin > np.pi) or (np.max(coadd_phis) - np.min(coadd_phis) > np.pi):
        #DEBUGGING - MARCM
        #print "Eps debugging"
        img_phis= np.mod( img_phis + np.pi, 2*np.pi )
        coadd_phis= np.mod( coadd_phis + np.pi, 2*np.pi )
        rangepix_phis = np.mod( rangepix_phis + np.pi, 2*np.pi )

    subweights = in_region(rangepix_thetas, rangepix_phis,
                                   img_thetas[ind_U], img_phis[ind_U], img_thetas[ind_L], img_phis[ind_L],
                                   img_thetas[ind_R], img_phis[ind_R], img_thetas[ind_B], img_phis[ind_B])
    # DEBUGGING - MARCM
    #print 'pmax pmin', pmax, pmin
    #print 'img_thetas again', img_thetas
    #print 'img_phis again', img_phis
    #print 'rangepix_phis', rangepix_phis
    #print 'rangepix_theta', rangepix_thetas
    #print 'subweights', subweights 
    
    if coadd_cut:
        subweights_coadd = in_region(rangepix_thetas, rangepix_phis,
                                   coadd_thetas[ind_U], coadd_phis[ind_U], coadd_thetas[ind_L], coadd_phis[ind_L],
                                   coadd_thetas[ind_R], coadd_phis[ind_R], coadd_thetas[ind_B], coadd_phis[ind_B])
        resubweights = np.logical_and(subweights, subweights_coadd).reshape(-1, nsubpixperpix)
    else:
        resubweights = subweights.reshape(-1, nsubpixperpix)

    sweights = resubweights.sum(axis=1) / float(nsubpixperpix)
    ind = (sweights > 0.0)
    
    # DEBUGGING - MARCM
    #print 'ind', ind
    #print 'ipixs_ring', ipixs_ring
    
    return ipixs_ring[ind], sweights[ind], img_thetas, img_phis, resubweights[ind,:]

def computeHPXpix_sequ_new_simp(nside, propertyArray): 
    #return 'ERROR'
    #Hack by AJR and MarcM, just return all of the pixel centers within the ra,dec range
    img_ras, img_decs = [propertyArray[v] for v in ['ra0', 'ra1', 'ra2','ra3']],[propertyArray[v] for v in ['dec0', 'dec1', 'dec2','dec3']]
    #print min(img_ras),max(img_ras)
    #more efficient version below failed for some reason
    #iweird = 0
    for i in range(0,len(img_ras)):
    	if img_ras[i] > 360.:
    		img_ras[i] -= 360.
    	if img_ras[i] < 0.:
    		img_ras[i] += 360.
    #if max(img_ras) - min(img_ras) > 1.:
    #	print img_ras,img_decs	
    #if np.any(img_ras > 360.0):
    #    img_ras[img_ras > 360.0] -= 360.0
    #if np.any(img_ras < 0.0):
    #    img_ras[img_ras < 0.0] += 360.0
    # Coordinates of image corners
    #print img_ras
    img_phis = np.multiply(img_ras , np.pi/180.)
    img_thetas =  np.pi/2.  - np.multiply(img_decs , np.pi/180.)
    img_pix = hp.ang2pix(nside, img_thetas, img_phis, nest=False)
    pix_thetas, pix_phis = hp.pix2ang(nside, img_pix, nest=False)
    ipix_list = np.zeros(0, dtype=int)
    # loop over rings until reached bottom
    iring_U = ring_num(nside, np.cos(img_thetas.min()), shift=0)
    iring_B = ring_num(nside, np.cos(img_thetas.max()), shift=0)
    ipixs_ring = []
    pmax = np.max(img_phis)
    pmin = np.min(img_phis)
    if pmax-pmin == 0:
    	return []
    p1 = pmin
    p2 = pmax

    if pmin < .1 and pmax > 1.9*np.pi:
	#straddling line
	#img_phis.sort()
        for i in range(0,len(img_phis)):
            if img_phis[i] > p1 and img_phis[i] < np.pi:
                p1 = img_phis[i]
            if img_phis[i] < p2 and img_phis[i] > np.pi:
                p2 = img_phis[i]
        #print 'kaka', img_phis, img_ras
        #print 'kaka', p1, p2, iring_U, iring_B	
        ipixs_ring1 = np.int64(np.concatenate([in_ring(nside, iring, 0, p1, conservative=False) for iring in range(iring_U, iring_B+1)]))
        ipixs_ring2 = np.int64(np.concatenate([in_ring(nside, iring, p2, 2.*np.pi, conservative=False) for iring in range(iring_U, iring_B+1)]))
 	#ipixs_ring1 = np.int64(np.concatenate([in_ring_simp(nside, iring, 0, p1, conservative=False) for iring in range(iring_U, iring_B+1)]))
 	#ipixs_ring2 = np.int64(np.concatenate([in_ring_simp(nside, iring, p2, 2.*np.pi, conservative=False) for iring in range(iring_U, iring_B+1)]))
        ipixs_ring = np.concatenate((ipixs_ring1,ipixs_ring2))
# 	print len(ipixs_ring),len(ipixs_ring1),len(ipixs_ring2),iring_B-iring_U,pmin,pmax,p1,p2
#     	
        if len(ipixs_ring1) > 1000: 
           print( 'kaka1', p1, iring_U, iring_B)
        if len(ipixs_ring2) > 1000:
           print( 'kaka2', p2, iring_U, iring_B)
    else:		
        ipixs_ring = np.int64(np.concatenate([in_ring(nside, iring, p1, p2, conservative=False) for iring in range(iring_U, iring_B+1)]))
	#ipixs_ring = np.int64(np.concatenate([in_ring_simp(nside, iring, p1, p2, conservative=False) for iring in range(iring_U, iring_B+1)]))
    if len(ipixs_ring) > 1000:
        #print 'hey', img_ras,img_decs 
    	print( 'careful', len(ipixs_ring),iring_B-iring_U,pmin,pmax,p1,p2)
        #nwrong = nwrong +1
    	return [] #temporary fix
    #	print len(ipixs_ring),iring_B-iring_U,pmin,pmax,min(img_ras),max(img_ras)  
    #print len(ipixs_ring),iring_B-iring_U,pmin,pmax,min(img_ras),max(img_ras)
    return ipixs_ring


# ---------------------------------------------------------------------------------------- #

# Crucial routine: read properties of a ccd image and returns its corners in ra dec.
# pixoffset is the number of pixels to truncate on the edges of each ccd image.
def computeCorners_WCS_TPV(propertyArray, pixoffset):
    #x = [1+pixoffset, propertyArray['NAXIS1']-pixoffset, propertyArray['NAXIS1']-pixoffset, 1+pixoffset, 1+pixoffset]
    #y = [1+pixoffset, 1+pixoffset, propertyArray['NAXIS2']-pixoffset, propertyArray['NAXIS2']-pixoffset, 1+pixoffset]
    x = [1+pixoffset, propertyArray['width']-pixoffset, propertyArray['width']-pixoffset, 1+pixoffset, 1+pixoffset]
    y = [1+pixoffset, 1+pixoffset, propertyArray['height']-pixoffset, propertyArray['height']-pixoffset, 1+pixoffset]
    #ras, decs = xy2radec(x, y, propertyArray)
    ras, decs = xy2radec_nopv(x, y, propertyArray)
    return ras, decs

# ---------------------------------------------------------------------------------------- #

# Performs WCS inverse projection to obtain ra dec from ccd image information.
def xy2radec(x, y, propertyArray):

    crpix = np.array( [ propertyArray['CRPIX1'], propertyArray['CRPIX2'] ] )
    cd = np.array( [ [ propertyArray['CD1_1'], propertyArray['CD1_2'] ],
                     [ propertyArray['CD2_1'], propertyArray['CD2_2'] ] ] )
    pv1 = [ float(propertyArray['PV1_'+str(k)]) for k in range(11) if k != 3 ] #  if k != 3
    pv2 = [ float(propertyArray['PV2_'+str(k)]) for k in range(11) if k != 3 ] #  if k != 3
    pv = np.array( [ [ [ pv1[0], pv1[2], pv1[5], pv1[9] ],
                                   [ pv1[1], pv1[4], pv1[8],   0.   ],
                                   [ pv1[3], pv1[7],   0.  ,   0.   ],
                                   [ pv1[6],   0.  ,   0.  ,   0.   ] ],
                                 [ [ pv2[0], pv2[1], pv2[3], pv2[6] ],
                                   [ pv2[2], pv2[4], pv2[7],   0.   ],
                                   [ pv2[5], pv2[8],   0.  ,   0.   ],
                                   [ pv2[9],   0.  ,   0.  ,   0.   ] ] ] )

    center_ra = propertyArray['CRVAL1'] * np.pi / 180.0
    center_dec = propertyArray['CRVAL2'] * np.pi / 180.0
    ras, decs = radec_gnom(x, y, center_ra, center_dec, cd, crpix, pv)
    ras = np.multiply( ras, 180.0 / np.pi )
    decs = np.multiply( decs, 180.0 / np.pi )
    if np.any(ras > 360.0):
        ras[ras > 360.0] -= 360.0
    if np.any(ras < 0.0):
        ras[ras < 0.0] += 360.0
    return ras, decs

def xy2radec_nopv(x, y, propertyArray):

    crpix = np.array( [ propertyArray['crpix1'], propertyArray['crpix2'] ] )
    cd = np.array( [ [ propertyArray['cd1_1'], propertyArray['cd1_2'] ],
                     [ propertyArray['cd2_1'], propertyArray['cd2_2'] ] ] )

    center_ra = propertyArray['crval1'] * np.pi / 180.0
    center_dec = propertyArray['crval2'] * np.pi / 180.0
    ras, decs = radec_gnom(x, y, center_ra, center_dec, cd, crpix, pv=False)
    ras = np.multiply( ras, 180.0 / np.pi )
    decs = np.multiply( decs, 180.0 / np.pi )
    if np.any(ras > 360.0):
        ras[ras > 360.0] -= 360.0
    if np.any(ras < 0.0):
        ras[ras < 0.0] += 360.0
    return ras, decs


# ---------------------------------------------------------------------------------------- #

# Deproject into ra dec values
def deproject_gnom(u, v, center_ra, center_dec):
    u *= arcsec_to_radians
    v *= arcsec_to_radians
    rsq = u*u + v*v
    cosc = sinc_over_r = 1./np.sqrt(1.+rsq)
    cosdec = np.cos(center_dec)
    sindec = np.sin(center_dec)
    sindec = cosc * sindec + v * sinc_over_r * cosdec
    tandra_num = -u * sinc_over_r
    tandra_denom = cosc * cosdec - v * sinc_over_r * sindec
    dec = np.arcsin(sindec)
    ra = center_ra + np.arctan2(tandra_num, tandra_denom)
    return ra, dec

# ---------------------------------------------------------------------------------------- #

def radec_gnom(x, y, center_ra, center_dec, cd, crpix, pv):
	p1 = np.array( [ np.atleast_1d(x), np.atleast_1d(y) ] )
	p2 = np.dot(cd, p1 - crpix[:,np.newaxis])
	u = p2[0]
	v = p2[1]
	if pv:
		usq = u*u
		vsq = v*v
		ones = np.ones(u.shape)
		upow = np.array([ ones, u, usq, usq*u ])
		vpow = np.array([ ones, v, vsq, vsq*v ])
		temp = np.dot(pv, vpow)
		p2 = np.sum(upow * temp, axis=1)
		u = - p2[0] * degree_to_arcsec
		v = p2[1] * degree_to_arcsec
	else:
		u = -u * degree_to_arcsec
		v = v * degree_to_arcsec
	ra, dec = deproject_gnom(u, v, center_ra, center_dec)
	return ra, dec

# ---------------------------------------------------------------------------------------- #

# Class for a pixel of the map, containing trees of images and values
class NDpix_simp:

    def __init__(self, propertyArray_in):
        self.nbelem = 1
        self.ratiores = 1
        self.propertyArray = [propertyArray_in]

    def addElem(self, propertyArray_in):
        self.nbelem += 1
        self.propertyArray.append(propertyArray_in)


    # Project NDpix into a single number
    # for a given property and operation applied to its array of images
    def project(self, property, weights, operation):

        asperpix = 0.263
        A = np.pi*(1.0/asperpix)**2
        pis = np.array([1.0 for proparr in self.propertyArray])


        # No super-resolution or averaging
        vals = np.array([proparr[property] for proparr in self.propertyArray])
        if operation == 'mean':
            return np.mean(vals)
        if operation == 'median':
            return np.median(vals)
        if operation == 'total':
            return np.sum(vals)
        if operation == 'min':
            return np.min(vals)
        if operation == 'max':
            return np.max(vals)
        if operation == 'maxmin':
            return np.max(vals) - np.min(vals)
        if operation == 'fracdet':
            return 1.0
        if operation == 'num':
            return len(vals)


# Class for a pixel of the map, containing trees of images and values
class NDpix:

    def __init__(self, propertyArray_in, inweights, ratiores):
        self.ratiores = ratiores
        self.nbelem = 1
        self.propertyArray = [propertyArray_in]
        if self.ratiores > 1:
            self.weights = np.array([inweights])

    def addElem(self, propertyArray_in, inweights):
        self.nbelem += 1
        self.propertyArray.append(propertyArray_in)
        if self.ratiores > 1:
            self.weights = np.vstack( (self.weights, inweights) )


    # Project NDpix into a single number
    # for a given property and operation applied to its array of images
    def project(self, property, weights, operation):

        asperpix = 0.263
        A = np.pi*(1.0/asperpix)**2
        # Computes COADD weights
        if weights == 'coaddweights3' or weights == 'coaddweights2' or weights == 'coaddweights' or property == 'maglimit2' or property == 'maglimit' or property == 'maglimit3' or property == 'sigmatot':
            m_zpi = np.array([proparr['MAGZP'] for proparr in self.propertyArray])
            if property == 'sigmatot':
                m_zp = np.array([30.0 for proparr in self.propertyArray])
            else:
                m_zp = np.array([proparr['COADD_MAGZP'] for proparr in self.propertyArray])
                
            if weights == 'coaddweights' or property == 'maglimit':
                sigma_bgi = np.array([
                    1.0/np.sqrt((proparr['WEIGHTA']+proparr['WEIGHTB'])/2.0)
                    if (proparr['WEIGHTA']+proparr['WEIGHTB']) >= 0.0 else proparr['SKYSIGMA']
                    for proparr in self.propertyArray])
            if weights == 'coaddweights2' or property == 'maglimit2':
                sigma_bgi = np.array([
                    0.5/np.sqrt(proparr['WEIGHTA'])+0.5/np.sqrt(proparr['WEIGHTB'])
                    if (proparr['WEIGHTA']+proparr['WEIGHTB']) >= 0.0 else proparr['SKYSIGMA']
                    for proparr in self.propertyArray])
            if weights == 'coaddweights3' or property == 'maglimit3' or property == 'sigmatot':
                sigma_bgi = np.array([proparr['SKYSIGMA'] for proparr in self.propertyArray])
            sigpis = 100**((m_zpi-m_zp)/5.0)
            mspis = (sigpis/sigma_bgi)**2.0
            pis = (sigpis/sigma_bgi)**2.0
        elif weights == 'invsqrtexptime':
            pis = np.array([ 1.0 / np.sqrt(proparr['EXPTIME']) for proparr in self.propertyArray])
        else:
            pis = np.array([1.0 for proparr in self.propertyArray])
            
        pis = np.divide(pis, pis.mean())

        # No super-resolution or averaging
        if self.ratiores == 1:
            if property == 'count':
                vals = np.array([1.0 for proparr in self.propertyArray])
            elif property == 'sigmatot':
                return np.sqrt(1.0 / mspis.sum())
            elif property == 'maglimit3' or property == 'maglimit2' or property == 'maglimit':
                sigma2_tot = 1.0 / mspis.sum()
                return np.mean(m_zp) - 2.5*np.log10(10*np.sqrt(A*sigma2_tot) )
            else:
                vals = np.array([proparr[property] for proparr in self.propertyArray])
                vals = vals * pis
            if operation == 'mean':
                return np.mean(vals)
            if operation == 'median':
                return np.median(vals)
            if operation == 'total':
                return np.sum(vals)
            if operation == 'min':
                return np.min(vals)
            if operation == 'max':
                return np.max(vals)
            if operation == 'maxmin':
                return np.max(vals) - np.min(vals)
            if operation == 'fracdet':
                return 1.0
            if operation == 'num':
                return len(vals)


        # Retrieve property array and apply operation (with super-resolution)
        if property == 'count':
            vals = np.array([1.0 for proparr in self.propertyArray])
        elif property == 'maglimit2' or property == 'maglimit' or property == 'maglimit3' or property == 'sigmatot':
            vals = (sigpis/sigma_bgi)**2
        else:
            #print property
            vals = np.array([proparr[property] for proparr in self.propertyArray])
            vals = vals * pis

        theweights = self.weights
        weightedarray = (theweights.T * vals).T
        counts = (theweights.T * pis).sum(axis=1)
        ind = counts > 0
        
        if property == 'maglimit' or property == 'maglimit2' or property == 'maglimit3':
            sigma2_tot =  1.0 / weightedarray.sum(axis=0)
            maglims = np.mean(m_zp) - 2.5*np.log10(10*np.sqrt(A*sigma2_tot) )
            return maglims[ind].mean()
        if property == 'sigmatot':
            sigma2_tot =  1.0 / weightedarray.sum(axis=0)
            return np.sqrt(sigma2_tot)[ind].mean()
        if operation == 'min':
            return np.min(vals)
        if operation == 'max':
            return np.max(vals)
        if operation == 'maxmin':
            return np.max(vals) - np.min(vals)
        if operation == 'mean':
            return (weightedarray.sum(axis=0) / counts)[ind].mean()
        if operation == 'median':
            return np.ma.median(np.ma.array(weightedarray, mask=np.logical_not(theweights)), axis=0)[ind].mean()
        if operation == 'total':
            return weightedarray.sum(axis=0)[ind].mean()
        if operation == 'fracdet':
            temp = weightedarray.sum(axis=0)
            return temp[ind].size / float(temp.size)
        if operation == 'num':
            return len(vals)

# ---------------------------------------------------------------------------------------- #

# Project NDpix into a value
def projectNDpix(args):
    pix, property, weights, operation = args
    if pix != 0:
        return pix.project(self, property, weights, operation)
    else:
        return hp.UNSEEN

# Create a "healtree", i.e. a set of pixels with trees of images in them.
def makeHealTree(args):
    samplename, nside, ratiores, pixoffset, tbdata = args
    treemap = HealTree(nside)
    verbcount = 1000
    count = 0
    start = time.time()
    duration = 0
    if(verbose): print( '>', samplename, ': starting tree making')
    for i, propertyArray in enumerate(tbdata):
        count += 1
        start_one = time.time()
        # DEBUGGING - MARCM
        #print "debugging i ", i
        treemap.addElem(propertyArray, ratiores, pixoffset)
        end_one = time.time()
        duration += float(end_one - start_one)
        if count == verbcount:
            if(verbose): print( '>', samplename, ': processed images', i-verbcount+1, '-', i+1, '(on '+str(len(tbdata))+') in %.2f' % duration, 'sec (~ %.3f' % (duration/float(verbcount)), 'per image)')
            count = 0
            duration = 0
    end = time.time()
    if(verbose): print('>', samplename, ': tree making took : %.2f' % float(end - start), 'sec for', len(tbdata), 'images')
    return treemap

def makeHealTree_simp(args):
	#hack by AJR
    samplename, nside, tbdata = args
    treemap = HealTree(nside)
    verbcount = 1000
    count = 0
    start = time.time()
    duration = 0
    if(verbose): print( '>', samplename, ': starting tree making')
    for i, propertyArray in enumerate(tbdata):
        count += 1
        start_one = time.time()
        treemap.addElem_simp(propertyArray)
        end_one = time.time()
        duration += float(end_one - start_one)
        if count == verbcount:
            if(verbose): print( '>', samplename, ': processed images', i-verbcount+1, '-', i+1, '(on '+str(len(tbdata))+') in %.2f' % duration, 'sec (~ %.3f' % (duration/float(verbcount)), 'per image)')
            count = 0
            duration = 0
    end = time.time()
    if(verbose): print( '>', samplename, ': tree making took : %.2f' % float(end - start), 'sec for', len(tbdata), 'images')
    return treemap


# ---------------------------------------------------------------------------------------- #

# Class for multi-dimensional healpix map that can be
# created and processed in parallel.
class HealTree:

    # Initialise and create array of pixels
    def __init__(self, nside):
        self.nside = nside
        self.npix = 12*nside**2
        self.pixlist = np.zeros(self.npix, dtype=object)

    # Process image and absorb its properties
    def addElem(self, propertyArray, ratiores, pixoffset):
        # Retrieve pixel indices
        ipixels, weights, thetas_c, phis_c, subpixrings = computeHPXpix_sequ_new(self.nside, propertyArray, pixoffset=pixoffset, ratiores=ratiores)
        # DEBUGGING - MARCM
        #print "deguging ipix addElem", ipixels
        # For each pixel, absorb image properties
        for ii, (ipix, weight) in enumerate(zip(ipixels, weights)):
            if self.pixlist[ipix] == 0:
                self.pixlist[ipix] = NDpix(propertyArray, subpixrings[ii,:], ratiores)
            else:
                self.pixlist[ipix].addElem(propertyArray, subpixrings[ii,:])

    def addElem_simp(self, propertyArray):
        #AJR hack
        # Retrieve non-conservative pixel indices, no oversampling, just the pixels with centers in the CCD
        ipixels = computeHPXpix_sequ_new_simp(self.nside, propertyArray)
        # For each pixel, absorb image properties
        #if ipixels == -1:
        #	return True
        #if len(i
        for ipix in ipixels:
            if self.pixlist[ipix] == 0:
                self.pixlist[ipix] = NDpix_simp(propertyArray)
            else:
                self.pixlist[ipix].addElem(propertyArray)


     # Project HealTree into partial Healpix map
     # for a given property and operation applied to its array of images
    def project_partial(self, property, weights, operation, pool=None):
        ind = np.where(self.pixlist != 0)
        pixel = np.arange(self.npix)[ind]
        verbcount = pixel.size / 10
        count = 0
        start = time.time()
        duration = 0
        signal = np.zeros(pixel.size)
        for i, pix in enumerate(self.pixlist[ind]):
            count += 1
            start_one = time.time()
            signal[i] = pix.project(property, weights, operation)
            end_one = time.time()
            duration += float(end_one - start_one)
            if count == verbcount:
                if(verbose): print( '>', property, weights, operation, ': processed pixels', i-verbcount+1, '-', i+1, '(on '+str(pixel.size)+') in %.1e' % duration, 'sec (~ %.1e' % (duration/float(verbcount)), 'per pixel)')
                count = 0
                duration = 0
        end = time.time()
        print( '> Projection', property, weights, operation, ' took : %.2f' % float(end - start), 'sec for', pixel.size, 'pixels')
        #signal = [pix.project(property, weights, operation) for pix in self.pixlist[ind]]
        return pixel, signal

     # Project HealTree into regular Healpix map
     # for a given property and operation applied to its array of images
    def project(self, property, weights, operation, pool=None):
        outmap = np.zeros(self.npix)
        outmap.fill(hp.UNSEEN)
        if pool is None:
            for ipix, pix in enumerate(self.pixlist):
                if pix != 0:
                    outmap[ipix] = pix.project(property, weights, operation)
        else:
            outmap = np.array( pool.map( projectNDpix, [ (pix, property, weights, operation) for pix in self.pixlist ] ) )
        return outmap

# ---------------------------------------------------------------------------------------- #

def makeHpxMap(args):
    healtree, property, weights, operation = args
    return healtree.project(property, weights, operation)

# ---------------------------------------------------------------------------------------- #

def makeHpxMap_partial(args):
    healtree, property, weights, operation = args
    return healtree.project_partial(property, weights, operation)

# ---------------------------------------------------------------------------------------- #

def addElemHealTree(args):
    healTree, propertyArray, ratiores = args
    healTree.addElem(propertyArray, ratiores)

# ---------------------------------------------------------------------------------------- #

# Process image and absorb its properties
def addElem(args):
    iarr, tbdatadtype, propertyArray, nside, propertiesToKeep, ratiores = args
    propertyArray.dtype = tbdatadtype
    if(verbose): print( 'Processing image', iarr, propertyArray['RA'])
    # Retrieve pixel indices
    ipixels, weights, thetas_c, phis_c = computeHPXpix_sequ_new(nside, propertyArray, pixoffset=pixoffset, ratiores=ratiores)
    print( 'Processing image', iarr, thetas_c, phis_c)
    # For each pixel, absorb image properties
    for ipix, weight in zip(ipixels, weights):
        if globalTree[ipix] == 0:
            globalTree[ipix] = NDpix(propertyArray, propertiesToKeep, weight=weight)
        else:
            globalTree[ipix].addElem(propertyArray, propertiesToKeep, weight=weight)

# ---------------------------------------------------------------------------------------- #

# Read and project a Healtree into Healpix maps, and write them.
def project_and_write_maps(mode, propertiesweightsoperations, tbdata, catalogue_name, outrootdir, sample_names, inds, nside, ratiores, pixoffset, nsidesout=None):

    resol_prefix = 'nside'+str(nside)+'_oversamp'+str(ratiores)
    outroot = outrootdir + '/' + catalogue_name + '/' + resol_prefix + '/'
    mkdir_p(outroot)
    if mode == 1: # Fully sequential
        for sample_name, ind in zip(sample_names, inds):
            #print len(tbdata[ind]['ra1'])
            #plt.plot(tbdata[ind]['ra1'],tbdata[ind]['dec1'],'k,')
            #plt.show()
            treemap = makeHealTree( (catalogue_name+'_'+sample_name, nside, ratiores, pixoffset, np.array(tbdata[ind])) )
            for property, weights, operation in propertiesweightsoperations:
                cutmap_indices, cutmap_signal = makeHpxMap_partial( (treemap, property, weights, operation) )
                if nsidesout is None:
                    fname = outroot + '_'.join([catalogue_name, sample_name, resol_prefix, property, weights, operation]) + '.fits'
                    print( 'Creating and writing', fname)
                    write_partial_map(fname, cutmap_indices, cutmap_signal, nside, nest=False)
                else:
                    cutmap_indices_nest = hp.ring2nest(nside, cutmap_indices)
                    outmap_hi = np.zeros(hp.nside2npix(nside))
                    outmap_hi.fill(0.0) #outmap_hi.fill(hp.UNSEEN)
                    outmap_hi[cutmap_indices_nest] = cutmap_signal
                    for nside_out in nsidesout:
                        if nside_out == nside:
                            outmap_lo = outmap_hi
                        else:
                            outmap_lo = hp.ud_grade(outmap_hi, nside_out, order_in='NESTED', order_out='NESTED')
                        resol_prefix2 = 'nside'+str(nside_out)+'from'+str(nside)+'o'+str(ratiores)
                        outroot2 = outrootdir + '/' + catalogue_name + '/' + resol_prefix2 + '/'
                        mkdir_p(outroot2)
                        fname = outroot2 + '_'.join([catalogue_name, sample_name, resol_prefix2, property, weights, operation]) + '.fits'
                        print( 'Writing', fname)
                        hp.write_map(fname, outmap_lo, nest=True)
                        subprocess.call("gzip -f "+fname,shell=True)


    if mode == 3: # Fully parallel
        pool = Pool(len(inds))
        print( 'Creating HealTrees')
        treemaps = pool.map( makeHealTree,
                         [ (catalogue_name+'_'+samplename, nside, ratiores, pixoffset, np.array(tbdata[ind]))
                           for samplename, ind in zip(sample_names, inds) ] )

        for property, weights, operation in propertiesweightsoperations:
            print( 'Making maps for', property, weights, operation)
            outmaps = pool.map( makeHpxMap_partial,
                            [ (treemap, property, weights, operation) for treemap in treemaps ] )
            for sample_name, outmap in zip(sample_names, outmaps):
                fname = outroot + '_'.join([catalogue_name, sample_name, resol_prefix, property, weights, operation]) + '.fits'
                print( 'Writing', fname)
                cutmap_indices, cutmap_signal = outmap
                write_partial_map(fname, cutmap_indices, cutmap_signal, nside, nest=False)


    if mode == 2:  # Parallel tree making and sequential writing
        pool = Pool(len(inds))
        print( 'Creating HealTrees')
        treemaps = pool.map( makeHealTree,
                         [ (catalogue_name+'_'+samplename, nside, ratiores, pixoffset, np.array(tbdata[ind]))
                           for samplename, ind in zip(sample_names, inds) ] )

        for property, weights, operation in propertiesweightsoperations:
            for sample_name, treemap in zip(sample_names, treemaps):
                fname = outroot + '_'.join([catalogue_name, sample_name, resol_prefix, property, weights, operation]) + '.fits'
                print('Writing', fname)
                #outmap = makeHpxMap( (treemap, property, weights, operation) )
                #hp.write_map(fname, outmap, nest=False)
                cutmap_indices, cutmap_signal = makeHpxMap_partial( (treemap, property, weights, operation) )
                write_partial_map(fname, cutmap_indices, cutmap_signal, nside, nest=False)

def project_and_write_maps_simp(mode, propertiesweightsoperations, tbdata, catalogue_name, outrootdir, sample_names, inds, nside):
	#hack by AJR and MarcM
        #nwrong = 0 #number of wrong projected pixels
	resol_prefix = 'nside'+str(nside)+'_oversamp1'
	outroot = outrootdir + '/' + catalogue_name + '/' + resol_prefix + '/'
	mkdir_p(outroot)
	for sample_name, ind in zip(sample_names, inds):
		treemap = makeHealTree_simp( (catalogue_name+'_'+sample_name, nside, np.array(tbdata[ind])) )
		for property, weights, operation in propertiesweightsoperations:
			cutmap_indices, cutmap_signal = makeHpxMap_partial( (treemap, property, weights, operation) )
			fname = outroot + '_'.join([catalogue_name, sample_name, resol_prefix, property, weights, operation]) + '.fits'
			print('Creating and writing', fname)
			write_partial_map(fname, cutmap_indices, cutmap_signal, nside, nest=False)

        #print "number of wrong projected ccd-pointings is: ", nwrong 

# ---------------------------------------------------------------------------------------- #





def test():
	fname = '/Users/bl/Dropbox/Projects/Quicksip/data/SVA1_COADD_ASTROM_PSF_INFO.fits'
	#fname = '/Users/bl/Dropbox/Projects/Quicksip/data/Y1A1_IMAGEINFO_and_COADDINFO.fits'
	pixoffset = 10
	hdulist = pyfits.open(fname)
	tbdata = hdulist[1].data
	hdulist.close()
	nside = 1024
	ratiores = 4
	treemap = HealTree(nside)
	#results = pool.map(treemap.addElem, [imagedata for imagedata in tbdata])
	print( tbdata.dtype)
	#ind = np.ndarray([0])
	ind = np.where( tbdata['band'] == 'i' )
	import numpy.random
	ind = numpy.random.choice(ind[0], 1 )
	print( 'Number of images :', len(ind))
	hpxmap = np.zeros(hp.nside2npix(nside))
	ras_c = []
	decs_c = []
	for i, propertyArray in enumerate(tbdata[ind]):
	    ras_c.append(propertyArray['RA'])
	    decs_c.append(propertyArray['DEC'])
	plt.figure()
	for i, propertyArray in enumerate(tbdata[ind]):
	    print(i)
	    propertyArray.dtype = tbdata.dtype
	    listpix, weights, thetas_c, phis_c, listpix_sup = computeHPXpix_sequ_new(nside, propertyArray, pixoffset=pixoffset, ratiores=ratiores)
	    #listpix2, weights2, thetas_c2, phis_c2 = computeHPXpix_sequ(nside, propertyArray, pixoffset=pixoffset, ratiores=ratiores)
	    hpxmap = np.zeros(hp.nside2npix(nside))
	    hpxmap[listpix] = weights
	    hpxmap_sup = np.zeros(hp.nside2npix(ratiores*nside))
	    hpxmap_sup[listpix_sup] = 1.0
	    listpix_hi, weights_hi, thetas_c_hi, phis_c_hi, superind_hi = computeHPXpix_sequ_new(ratiores*nside, propertyArray, pixoffset=pixoffset, ratiores=1)
	    hpxmap_hi = np.zeros(hp.nside2npix(ratiores*nside))
	    hpxmap_hi[listpix_hi] = weights_hi
	    hpxmap_hitolo = hp.ud_grade(hpxmap_hi, nside)
	    print('valid hpxmap_hi', np.where(hpxmap_hi > 0)[0])
	    print('hpxmap', zip(np.where(hpxmap > 0)[0], hpxmap[hpxmap > 0]))
	    print('hpxmap_sup', zip(np.where(hpxmap_sup > 0)[0], hpxmap_sup[hpxmap_sup > 0]))
	    print('hpxmap_hitolo', zip(np.where(hpxmap_hitolo > 0)[0], hpxmap_hitolo[hpxmap_hitolo > 0]))
	    hp.gnomview(hpxmap_hi, title='hpxmap_hi', rot=[propertyArray['RA'], propertyArray['DEC']], reso=0.2)
	    hp.gnomview(hpxmap_sup, title='hpxmap_sup', rot=[propertyArray['RA'], propertyArray['DEC']], reso=0.2)
	    hp.gnomview(hpxmap_hitolo, title='hpxmap_hitolo', rot=[propertyArray['RA'], propertyArray['DEC']], reso=0.2)
	    hp.gnomview(hpxmap, title='hpxmap', rot=[propertyArray['RA'], propertyArray['DEC']], reso=0.2)
	    #plt.plot(phis_c, thetas_c)
	    thetas, phis = hp.pix2ang(nside, listpix)
	    #plt.scatter(phis, thetas, color='red', marker='o', s=50*weights)
	    #plt.scatter(propertyArray['RA']*np.pi/180, np.pi/2 - propertyArray['DEC']*np.pi/180)
	    #plt.text(propertyArray['RA']*np.pi/180, np.pi/2 - propertyArray['DEC']*np.pi/180, str(i))
	plt.show()
	stop

#if __name__ == "__main__":
#    test()
