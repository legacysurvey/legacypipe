from math import *
from legendre import legendre
twopi = 2.*pi
piover2 = .5*pi
ns_max = 8192
ee = exp(1.)


def pix2ang_ring(nside,ipix):

#c=======================================================================
#c     gives theta and phi corresponding to pixel ipix (RING) 
#c     for a parameter nside
#c=======================================================================


#		pi = 3.1415926535897932384626434
  
	ns_max=8192
	npix = 12*nside*nside


	ipix1 = ipix + 1
	nl2 = 2*nside
	nl4 = 4*nside
	ncap = 2*nside*(nside-1)#// ! points in each polar cap, =0 for nside =1
	fact1 = 1.5*nside
	fact2 = 3.0*nside*nside
	
	if ipix1 <= ncap :# {  //! North Polar cap -------------
		hip   = ipix1/2.
		fihip = floor(hip)
		iring = int(floor( sqrt( hip - sqrt(fihip) ) ) + 1)#;// ! counted from North pole
		iphi  = ipix1 - 2*iring*(iring - 1)
		
		theta = acos( 1. - iring*iring / fact2 )
		phi   = (1.*iphi - 0.5) * pi/(2.*iring)

	else:
		if ipix1 <= nl2*(5*nside+1):# {//then ! Equatorial region ------

			ip    = ipix1 - ncap - 1
			iring = int(floor( ip / nl4 ) + nside)#;// ! counted from North pole
			iphi  = int(fmod(ip,nl4) + 1)
			
			fodd  = 0.5 * (1 + fmod(float(iring+nside),2))#//  ! 1 if iring+nside is odd, 1/2 otherwise
			theta = acos( (nl2 - iring) / fact1 )
			phi   = (1.*iphi - fodd) * pi /(2.*nside)
		else:# {//! South Polar cap -----------------------------------

			ip    = npix - ipix1 + 1
			hip   = ip/2.
#/* bug corrige floor instead of 1.* */
			fihip = floor(hip)
			iring = int(floor( sqrt( hip - sqrt(fihip) ) ) + 1)#;//     ! counted from South pole
			iphi  = int((4.*iring + 1 - (ip - 2.*iring*(iring-1))))
			
			theta = acos( -1. + iring*iring / fact2 )
			phi   = (1.*iphi - 0.5) * pi/(2.*iring)
	
	return theta,phi

def ang2pix_ring(nside,theta,phi):
#    c=======================================================================
#   c     gives the pixel number ipix (RING) 
#    c     corresponding to angles theta and phi
#    c=======================================================================
  
	z0=2.0/3.0
	ns_max=8192
  
	z = cos(theta)
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
    
		ir = nside + 1 + jp - jm#;// ! in {1,2n+1} (ring number counted from z=2/3)
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
		
		ir = jp + jm + 1#;//        ! ring number counted from the closest pole
		ip = int(floor( tt * ir ) + 1)#;// ! in {1,4*ir}
		if ip>4*ir:
			ip = ip - 4*ir
		
		ipix1 = 2*ir*(ir-1) + ip
		if z<=0.:
			ipix1 = npix - 2*ir*(ir+1) + ip

	return ipix1 - 1



def healRDfullsky(res):
	#write out positions of pixel centers, for, e.g., plotting purposes
	np = 12*res*res
	h = healpix()
	fo = open('healRDfullsky'+str(res)+'.dat','w')
	for i in range(0,np):
		th,phi = h.pix2ang_nest(res,i)
		ra,dec = thphi2radec(th,phi)
		fo.write(str(ra)+' '+str(dec)+'\n')
	fo.close()
	return True



def pixl_up(file,reso,resn):
	h = healpix()
	f = open(file+str(reso)+'.dat')
	ol = []
	for line in f:
		ol.append(float(line))
	nl = []
	np = 12*resn*resn
	for i in range(0,np):
		nl.append(0)
	for i in range(0,len(ol)):
		th,phi = h.pix2ang_nest(reso,i)
		p = int(h.ang2pix_nest(resn,th,phi))
		nl[p] += ol[i]
	fo = open(file+str(resn)+'.dat','w')
	for i in range(0,np):
		fo.write(str(nl[i])+'\n')
	fo.close()
	return True

def ranpixl_up(file,reso,resn):
	h = healpix()
	f = open('ranHeal_pix'+str(reso)+file+'.dat')
	ol = []
	npo = 12*reso*reso
	for i in range(0,npo):
		ol.append(0)
	for line in f:
		ln = line.split()
		p = int(ln[0])
		ol[p] += float(ln[1])
	nl = []
	np = 12*resn*resn
	for i in range(0,np):
		nl.append(0)
	for i in range(0,len(ol)):
		th,phi = h.pix2ang_nest(reso,i)
		p = int(h.ang2pix_nest(resn,th,phi))
		nl[p] += ol[i]
	fo = open('ranHeal_pix'+str(resn)+file+'.dat','w')
	for i in range(0,np):
		if nl[i] != 0:
			fo.write(str(i)+' '+str(nl[i])+'\n')
	fo.close()
	return True


def mkhealpixl_simp(file,res=256,rc=0,dc=1,zc=2,md='csv'):
	h = healpix()
	pixl = []
	npo = 12*res**2
	for i in range(0,npo):
		pixl.append(0)
	
	f = open(file+'.'+md)
	f.readline()
	n = 0
	for line in f:
		if line[0] != '#':
			if md == 'csv':
				ln = line.split(',')
			else:
				ln = line.split()
			try:
				ra,dec = float(ln[rc]),float(ln[dc])
				th,phi = radec2thphi(ra,dec)
				p = int(h.ang2pix_nest(res,th,phi))
				pixl[p] += 1.
				n += 1
			except:
				pass
	print n
	fo = open(file+'hpixall'+str(res)+'.dat','w')
	for i in range(0,npo):
		th,phi = h.pix2ang_nest(res,i)
		fo.write(str(pixl[i])+'\n')
	fo.close()
	return True


def thphi2le(theta,phi):
	deg2Rad = pi/180.0
	rarad = phi
	decrad = -(theta-piover2)
	surveyCenterDEC = 32.5
	surveyCenterRA = 185.0
	etaPole = deg2Rad*surveyCenterDEC
	node = deg2Rad*(surveyCenterRA - 90.0)
	
	x = cos(rarad-node)*cos(decrad)
  	y = sin(rarad-node)*cos(decrad)
  	z = sin(decrad)

  	lam = -1.0*asin(x)/deg2Rad
  	eta = (atan2(z,y) - etaPole)/deg2Rad
  	if eta < -180.0:
  		eta += 360.0
  	if eta > 180.0:
  		eta -= 360.0
  	
  	return (lam,eta)

def le2thetaphi(lam,eta):
	deg2Rad = pi/180.0
	surveyCenterDEC = 32.5
	surveyCenterRA = 185.0
	etaPole = deg2Rad*surveyCenterDEC
	node = deg2Rad*(surveyCenterRA - 90.0)
	x = -1.0*sin(lam*deg2Rad)
	y = cos(lam*deg2Rad)*cos(eta*deg2Rad+etaPole)
	z = cos(lam*deg2Rad)*sin(eta*deg2Rad+etaPole)
	ra = (atan2(y,x) + node)
	if ra < 0.0:
		ra +=twopi
	dec = asin(z)
	return -dec+piover2,ra




def radec2thphi(ra,dec):
	return (-dec+90.)*pi/180.,ra*pi/180.

def thphi2radec(theta,phi):
	return 180./pi*phi,-(180./pi*theta-90)


class healpix:
	#translated from c by Ashley J. Ross; no guarantees but everything has checked out so far... 
	def __init__(self):
		self.pix2x,self.pix2y = self.mk_pix2xy()
		self.x2pix,self.y2pix = self.mk_xy2pix()

	def ang2pix_nest(self,nside,theta,phi):
		#x2pix,y2pix = mk_xy2pix()
		z  = cos(theta)
		za = fabs(z)
		z0 = 2./3.
		if phi>=twopi:
			phi = phi - twopi
		if phi<0.:
			phi = phi + twopi
		tt = phi / piover2
		if za<=z0:  #{ /* equatorial region */
		
		#/* (the index of edge lines increase when the longitude=phi goes up) */
			jp = int(floor(ns_max*(0.5 + tt - z*0.75)))# /* ascending edge line index */
			jm = int(floor(ns_max*(0.5 + tt + z*0.75)))#; /* descending edge line index */
			
			#/* finds the face */
			ifp = jp / ns_max#; /* in {0,4} */
			ifm = jm / ns_max
			
			if ifp==ifm:
				face_num = int(fmod(ifp,4)) + 4#; /* faces 4 to 7 */
			else:
				if ifp<ifm:
					face_num = int(fmod(ifp,4)) #/* (half-)faces 0 to 3 */
				else:
					face_num = int(fmod(ifm,4)) + 8#;           /* (half-)faces 8 to 11 */
			
			ix = int(fmod(jm, ns_max))
			iy = ns_max - int(fmod(jp, ns_max)) - 1
	
		else:# { /* polar region, za > 2/3 */
		
			ntt = int(floor(tt))
			if ntt>=4:
				ntt = 3
			tp = tt - ntt
			tmp = sqrt( 3.*(1. - za) )#; /* in ]0,1] */
		
		#/* (the index of edge lines increase when distance from the closest pole
		# * goes up)
		# */
		#/* line going toward the pole as phi increases */
			jp = int(floor( ns_max * tp * tmp )) 
		
		#/* that one goes away of the closest pole */
			jm = int(floor( ns_max * (1. - tp) * tmp ))
			if jp >= ns_max:
				jp = ns_max-1
			if jm >= ns_max:
				jm = ns_max-1
			#jp = int((jp < ns_max-1 ? jp : ns_max-1))
			#jm = (int)(jm < ns_max-1 ? jm : ns_max-1);
		
		#/* finds the face and pixel's (x,y) */
			if z>=0:# ) {
				face_num = ntt#; /* in {0,3} */
				ix = ns_max - jm - 1
				iy = ns_max - jp - 1
			else:
				face_num = ntt + 8#; /* in {8,11} */
				ix =  jp
				iy =  jm	
	
		ix_low = int(fmod(ix,128))
		ix_hi  = ix/128
		iy_low = int(fmod(iy,128))
		iy_hi  = iy/128
	
		ipf = (self.x2pix[ix_hi]+self.y2pix[iy_hi]) * (128 * 128)+ (self.x2pix[ix_low]+self.y2pix[iy_low]);
		ipf = (long)(ipf / pow(ns_max/nside,2))#;     /* in {0, nside**2 - 1} */
		return ( ipf + face_num*pow(nside,2))#; /* in {0, 12*nside**2 - 1} */
	
	def mk_xy2pix(self):
	#   /* =======================================================================
	#    * subroutine mk_xy2pix
	#    * =======================================================================
	#    * sets the array giving the number of the pixel lying in (x,y)
	#    * x and y are in {1,128}
	#    * the pixel number is in {0,128**2-1}
	#    *
	#    * if  i-1 = sum_p=0  b_p * 2^p
	#    * then ix = sum_p=0  b_p * 4^p
	#    * iy = 2*ix
	#    * ix + iy in {0, 128**2 -1}
	#    * =======================================================================
	#    */
	#  int i, K,IP,I,J,ID;
		x2pix = []
		y2pix = []
		for i in range(0,128):#(i = 0; i < 127; i++) x2pix[i] = 0;
			x2pix.append(0)
			y2pix.append(0)
		for I in range(1,129):#( I=1;I<=128;I++ ) {
			J  = I-1#;//            !pixel numbers
			K  = 0#;//
			IP = 1#;//
			while J!=0:
	#    truc : if( J==0 ) {
	#     x2pix[I-1] = K;
	#      y2pix[I-1] = 2*K;
	#    }
	#    else {
				ID = int(fmod(J,2))
				J  = J/2
				K  = IP*ID+K
				IP = IP*4
	#      goto truc;
			x2pix[I-1] = K
			y2pix[I-1] = 2*K
		return x2pix,y2pix
	
	def mk_pix2xy(self): 
	
	#   /* =======================================================================
	#    * subroutine mk_pix2xy
	#    * =======================================================================
	#    * constructs the array giving x and y in the face from pixel number
	#    * for the nested (quad-cube like) ordering of pixels
	#    *
	#    * the bits corresponding to x and y are interleaved in the pixel number
	#    * one breaks up the pixel number by even and odd bits
	#    * =======================================================================
	#    */
	
	#  int i, kpix, jpix, IX, IY, IP, ID;
		pix2x = []
		pix2y = []
		for i in range(0,1024):
			pix2x.append(0)
			pix2y.append(0)
	#  for (i = 0; i < 1023; i++) pix2x[i]=0;
	  
	#  for( kpix=0;kpix<1024;kpix++ ) {
		for kpix in range(0,1024):
			jpix = kpix
			IX = 0
			IY = 0
			IP = 1# ;//              ! bit position (in x and y)
			while jpix!=0:# ){// ! go through all the bits
				ID = int(fmod(jpix,2))#;//  ! bit value (in kpix), goes in ix
				jpix = jpix/2
				IX = ID*IP+IX
				
				ID = int(fmod(jpix,2))#;//  ! bit value (in kpix), goes in iy
				jpix = jpix/2
				IY = ID*IP+IY
				
				IP = 2*IP#;//         ! next bit (in x and y)
			pix2x[kpix] = IX#;//     ! in 0,31
			pix2y[kpix] = IY#;//     ! in 0,31
	  
		return pix2x,pix2y
	
	def pix2ang_nest(self,nside, ipix):
	
	#   /*
	#     c=======================================================================
	#     subroutine pix2ang_nest(nside, ipix, theta, phi)
	#     c=======================================================================
	#     c     gives theta and phi corresponding to pixel ipix (NESTED) 
	#     c     for a parameter nside
	#     c=======================================================================
	#   */
		
		#pix2x,pix2y = mk_pix2xy()
		jrll = []
		jpll = []
		for i in range(0,12):
			jrll.append(0)
			jpll.append(0)
		jrll[0]=2
		jrll[1]=2
		jrll[2]=2
		jrll[3]=2
		jrll[4]=3
		jrll[5]=3
		jrll[6]=3
		jrll[7]=3
		jrll[8]=4
		jrll[9]=4
		jrll[10]=4
		jrll[11]=4
		jpll[0]=1
		jpll[1]=3
		jpll[2]=5
		jpll[3]=7
		jpll[4]=0
		jpll[5]=2
		jpll[6]=4
		jpll[7]=6
		jpll[8]=1
		jpll[9]=3
		jpll[10]=5
		jpll[11]=7
		  
		  
		npix = 12 * nside*nside
		if ipix < 0 or ipix > npix-1:
			return 'ipix out of range'
	
	#      /* initiates the array for the pixel number -> (x,y) mapping */
	
		fn = 1.*nside
		fact1 = 1./(3.*fn*fn)
		fact2 = 2./(3.*fn)
		nl4   = 4*nside
	
	#      //c     finds the face, and the number in the face
		npface = nside*nside
		
		face_num = ipix/npface#//  ! face number in {0,11}
		ipf = int(fmod(ipix,npface))#//  ! pixel number in the face {0,npface-1}
		
	#	//c     finds the x,y on the face (starting from the lowest corner)
	#	//c     from the pixel number
		ip_low = int(fmod(ipf,1024))#;//       ! content of the last 10 bits
		ip_trunc =   ipf/1024# ;//       ! truncation of the last 10 bits
		ip_med = int(fmod(ip_trunc,1024))#;//  ! content of the next 10 bits
		ip_hi  =     ip_trunc/1024   #;//! content of the high weight 10 bits
		
		ix = 1024*self.pix2x[ip_hi] + 32*self.pix2x[ip_med] + self.pix2x[ip_low]
		iy = 1024*self.pix2y[ip_hi] + 32*self.pix2y[ip_med] + self.pix2y[ip_low]
		
	#	//c     transforms this in (horizontal, vertical) coordinates
		jrt = ix + iy#;//  ! 'vertical' in {0,2*(nside-1)}
		jpt = ix - iy#;//  ! 'horizontal' in {-nside+1,nside-1}
		
	#	//c     computes the z coordinate on the sphere
	#	//      jr =  jrll[face_num+1]*nside - jrt - 1;//   ! ring number in {1,4*nside-1}
		jr =  jrll[face_num]*nside - jrt - 1
	#	//      cout << "face_num=" << face_num << endl;
	#	//      cout << "jr = " << jr << endl;
	#	//      cout << "jrll(face_num)=" << jrll[face_num] << endl;
	#	//      cout << "----------------------------------------------------" << endl;
		nr = nside#;//                  ! equatorial region (the most frequent)
		z  = (2*nside-jr)*fact2
		kshift = int(fmod(jr - nside, 2))
		if jr<nside:#  { //then     ! north pole region
			nr = jr
			z = 1. - nr*nr*fact1
			kshift = 0
	
		else:# {
			if jr>3*nside:# {// then ! south pole region
				 nr = nl4 - jr
				 z = - 1. + nr*nr*fact1
				 kshift = 0
		theta = acos(z)
		
	#	//c     computes the phi coordinate on the sphere, in [0,2Pi]
	#	//      jp = (jpll[face_num+1]*nr + jpt + 1 + kshift)/2;//  ! 'phi' number in the ring in {1,4*nr}
		jp = (jpll[face_num]*nr + jpt + 1 + kshift)/2
		if jp>nl4:
			jp = jp - nl4
		if jp<1:
			jp = jp + nl4
		
		phi = (jp - (kshift+1)*0.5) * (piover2 / nr)
		return theta,phi

	def ring2nest(self,nside,p_ring): 
#	"""  /*
#		c=======================================================================
#		subroutine ring2nest(nside, ipring, ipnest)
#		c=======================================================================
#		c     conversion from RING to NESTED pixel number
#		c=======================================================================
#	  */
#	"""  
		ns_max=8192
	  
	#  static int x2pix[128], y2pix[128];
	#  //      common    /xy2pix/ x2pix,y2pix
	
		jrll = []
		jpll = []#;// ! coordinate of the lowest corner of each face
		for i in range(0,12):
			jrll.append(0)
			jpll.append(0)
		jrll[0]=2
		jrll[1]=2
		jrll[2]=2
		jrll[3]=2
		jrll[4]=3
		jrll[5]=3
		jrll[6]=3
		jrll[7]=3
		jrll[8]=4
		jrll[9]=4
		jrll[10]=4
		jrll[11]=4
		jpll[0]=1
		jpll[1]=3
		jpll[2]=5
		jpll[3]=7
		jpll[4]=0
		jpll[5]=2
		jpll[6]=4
		jpll[7]=6
		jpll[8]=1
		jpll[9]=3
		jpll[10]=5
		jpll[11]=7
	  
		npix = 12 * nside*nside
	#  if( ipring<0 || ipring>npix-1 ) {
	#    fprintf(stderr, "ipring out of range\n");
	#    exit(0);
	#  }
		if x2pix[127]<=0:
			self.mk_xy2pix()
	  
		nl2 = 2*nside
		nl4 = 4*nside
		npix = 12*nside*nside#;//      ! total number of points
		ncap = 2*nside*(nside-1)#;// ! points in each polar cap, =0 for nside =1
		ipring1 = p_ring + 1
	  
	#  //c     finds the ring number, the position of the ring and the face number
		if ipring1<=ncap: #//then
		
			hip   = ipring1/2.
			fihip = int(floor ( hip ))
			irn   = int(floor( sqrt( hip - sqrt(fihip) ) ) + 1)#;// ! counted from North pole
			iphi  = ipring1 - 2*irn*(irn - 1);
			
			kshift = 0
			nr = irn  # ;//               ! 1/4 of the number of points on the current ring
			face_num = (iphi-1) / irn#;// ! in {0,3}
	
		else:
			if ipring1<=nl2*(5*nside+1):# {//then
		
				ip    = ipring1 - ncap - 1
				irn   = int(floor( ip / nl4 ) + nside)#;//               ! counted from North pole
				iphi  = int(fmod(ip,nl4) + 1)
				
				kshift  = int(fmod(irn+nside,2))#;//  ! 1 if irn+nside is odd, 0 otherwise
				nr = nside
				ire =  irn - nside + 1#;// ! in {1, 2*nside +1}
				irm =  nl2 + 2 - ire
				ifm = (iphi - ire/2 + nside -1) / nside#;// ! face boundary
				ifp = (iphi - irm/2 + nside -1) / nside
				if ifp==ifm:# {//then          ! faces 4 to 7
					face_num = int(fmod(ifp,4) + 4)
				
				else:
					if ifp + 1==ifm:#  {//then ! (half-)faces 0 to 3
						face_num = ifp
				
					else:
						if ifp - 1==ifm:# {//then ! (half-)faces 8 to 11
							face_num = ifp + 7
	 
	
			else:
		
				ip    = npix - ipring1 + 1
				hip   = ip/2.
				fihip = floor ( hip )
				irs   = int(floor( sqrt( hip - sqrt(fihip) ) ) + 1)#;//  ! counted from South pole
				iphi  = 4*irs + 1 - (ip - 2*irs*(irs-1))
				
				kshift = 0
				nr = irs
				irn   = nl4 - irs
				face_num = (iphi-1) / irs + 8#;// ! in {8,11}
	  
	#  //c     finds the (x,y) on the face
	#  //  irt =   irn  - jrll[face_num+1]*nside + 1;//       ! in {-nside+1,0}
	#  //  ipt = 2*iphi - jpll[face_num+1]*nr - kshift - 1;// ! in {-nside+1,nside-1}
		irt =   irn  - jrll[face_num]*nside + 1#;//       ! in {-nside+1,0}
		ipt = 2*iphi - jpll[face_num]*nr - kshift - 1
	
	
		if ipt>=nl2:
			ipt = ipt - 8*nside#;// ! for the face #4
	  
		ix =  (ipt - irt ) / 2
		iy = -(ipt + irt ) / 2
		
		ix_low = int(fmod(ix,128))
		ix_hi  = ix/128
		iy_low = int(fmod(iy,128))
		iy_hi  = iy/128
	#  //  cout << "ix_low = " << ix_low << " ix_hi = " << ix_hi << endl;
	#  //  cout << "iy_low = " << iy_low << " iy_hi = " << iy_hi << endl;
	#  //  ipf =  (x2pix[ix_hi +1]+y2pix[iy_hi +1]) * (128 * 128)
	#  //    + (x2pix[ix_low+1]+y2pix[iy_low+1]);//        ! in {0, nside**2 - 1}
		ipf =  (x2pix[ix_hi]+y2pix[iy_hi]) * (128 * 128)+ (x2pix[ix_low]+y2pix[iy_low])
	
	
	#  //  cout << "ipf = " << ipf << endl;
	#  //  for( int i(0);i<128;i++ ) cout << x2pix[i] << " || " << y2pix[i] << endl;
		return ipf + face_num* nside *nside#;//   ! in {0, 12*nside**2 - 1}
	  




