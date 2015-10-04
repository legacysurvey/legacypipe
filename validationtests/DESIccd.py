dir = '$HOME/' # obviously needs to be changed
from math import *

### A couple of useful conversions

def zeropointToScale(zp):
	return 10.**((zp - 22.5)/2.5)	

def nanomaggiesToMag(nm):
	return -2.5 * (log(nm,10.) - 9.)

### Find calculate depth stats in magnitudes

def plotMaghist(band,nbin=100):
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	readnoise = 10. # e-; 7.0 to 15.0 according to DECam Data Handbook
	p = 1.15 #value given in imaging requirements
	gain = 4.0 #from Dustin
	f = fitsio.read(dir+'/legacypipe-dir/decals-ccds.fits.gz')
	NTl = []
	emin = 1000
	emax = 0
	msee = 0
	n = 0
	arcsec2pix = 3.6 #this number is just taken from the ratio between fwhm and seeing and might not be correct!
	for i in range(0,len(f)):
		if f[i]['filter'] == band:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and f[i]['exptime'] >19:   
				avsky = f[i]['avsky']
				skysig = sqrt(avsky * gain + readnoise**2) / gain
				zpscale = zeropointToScale(f[i]['ccdzpt'] + 2.5*log(f[i]['exptime'],10.))
				skysig /= zpscale
				psf_sigma = f[i]['fwhm'] / 2.35
				# point-source depth
				#psfnorm = 1./(2. * sqrt(pi) * psf_sigma) #1/Neff #for point source
				#detsig1 = skysig / psfnorm
				Np = ((4.*pi*psf_sigma**2.)**(1./p) + (8.91*(.45*arcsec2pix)**2. )**(1./p))**p #Neff in requirements doc
				Np = sqrt(Np) #square root necessary because Np gives sum of noise squared
				detsig1 = skysig*Np #total noise
				m = nanomaggiesToMag(detsig1 * 5.)
				if m > 30 or m < 18:
					print skysig,avsky,f[i]['fwhm'],f[i]['ccdzpt'],f[i]['exptime']
				NTl.append(m)
				if f[i]['exptime'] > emax:
					emax = f[i]['exptime']
				if 	f[i]['exptime'] < emin:
					emin = f[i]['exptime']
				n += 1.
				msee += f[i]['seeing']	
	msee = msee/n
	print msee,n
	print emin,emax
	minN = min(NTl)
	maxN = max(NTl)+.0001
	print minN,maxN
	hl = zeros((nbin))
	for i in range(0,len(NTl)):
		bin = int(nbin*(NTl[i]-minN)/(maxN-minN))
		hl[bin] += 1
	Nl = []
	for i in range(0,len(hl)):
		Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))
	NTl = array(NTl)
	mean = sum(NTl)/float(len(NTl))
	std = sqrt(sum(NTl**2.)/float(len(NTl))-mean**2.)
	NTl.sort()
	if len(NTl)/2. != len(NTl)/2:
		med = NTl[len(NTl)/2+1]
	else:
		med = (NTl[len(NTl)/2+1]+NTl[len(NTl)/2])/2.
	print mean,med,std
	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title(str(mean)[:5]+r'$\pm$'+str(std)[:4])
	#plt.xscale('log')
	plt.show()
	return True

###Below are obsolete modules used for testing

def plotNeffThist(band,nbin=100):
	#plots the Neff quantity, defined in the imagining requirements, multiplied by the exposure time
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	p = 1.15
	f = fitsio.read(dir+'/legacypipe-dir/decals-ccds.fits.gz')
	NTl = []
	emin = 1000
	emax = 0
	for i in range(0,len(f)):
		if f[i]['filter'] == band:
			if f[i]['seeing'] != 99:
				NT = f[i]['exptime']/((4.*pi*(f[i]['fwhm'])**2.)**(1./p) + (8.91*.45**2. )**(1./p))**p
				NTl.append(NT)
				if f[i]['exptime'] > emax:
					emax = f[i]['exptime']
				if 	f[i]['exptime'] < emin:
					emin = f[i]['exptime']
				if NT > 5.e6:
					print i,f[i]['exptime'],f[i]['seeing']
	print emin,emax
	minN = min(NTl)
	maxN = max(NTl)+.0001
	print minN,maxN
	hl = zeros((nbin))
	for i in range(0,len(NTl)):
		bin = int(nbin*(NTl[i]-minN)/(maxN-minN))
		hl[bin] += 1
	Nl = []
	for i in range(0,len(hl)):
		Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))
	NTl = array(NTl)
	mean = sum(NTl)/float(len(NTl))
	std = sqrt(sum(NTl**2.)/float(len(NTl))-mean**2.)
	print mean,std
	plt.plot(Nl,hl,'k-')
	#plt.xscale('log')
	plt.show()
	return True

def plotIVARhist(band,nbin=100):
	#plots the Neff quantity, defined in the imagining requirements, multiplied by the exposure time
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	p = 1.15
	f = fitsio.read(dir+'/legacypipe-dir/decals-ccds.fits.gz')
	NTl = []
	emin = 1000
	emax = 0
	for i in range(0,len(f)):
		if f[i]['filter'] == band:
			if f[i]['seeing'] != 99:
				NT = f[i]['avsky']*((4.*pi*(f[i]['fwhm'])**2.)**(1./p) + (8.91*.45**2. )**(1./p))**p/f[i]['exptime']
				NTl.append(NT)
				if f[i]['exptime'] > emax:
					emax = f[i]['exptime']
				if 	f[i]['exptime'] < emin:
					emin = f[i]['exptime']
				if NT > 5.e6:
					print i,f[i]['exptime'],f[i]['seeing']
	print emin,emax
	minN = min(NTl)
	maxN = max(NTl)+.0001
	print minN,maxN
	hl = zeros((nbin))
	for i in range(0,len(NTl)):
		bin = int(nbin*(NTl[i]-minN)/(maxN-minN))
		hl[bin] += 1
	Nl = []
	for i in range(0,len(hl)):
		Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))
	NTl = array(NTl)
	mean = sum(NTl)/float(len(NTl))
	std = sqrt(sum(NTl**2.)/float(len(NTl))-mean**2.)
	print mean,std
	plt.plot(Nl,hl,'k-')
	#plt.xscale('log')
	plt.show()
	return True


		
		