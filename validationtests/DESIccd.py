dir = '$HOME/' # obviously needs to be changed
from math import *

### A couple of useful conversions

def zeropointToScale(zp):
	return 10.**((zp - 22.5)/2.5)	

def nanomaggiesToMag(nm):
	return -2.5 * (log(nm,10.) - 9.)

'''
* CCDNMATCH >= 20 (At least 20 stars to determine zero-pt)
* abs(ZPT - CCDZPT) < 0.10  (Loose agreement with full-frame zero-pt)
* ZPT within [25.08-0.50, 25.08+0.25] for g-band
* ZPT within [25.29-0.50, 25.29+0.25] for r-band
* ZPT within [24.92-0.50, 24.92+0.25] for z-band
* DEC > -20 (in DESI footprint)
* EXPTIME >= 30
* CCDNUM = 31 (S7) should mask outside the region [1:1023,1:4094]
'''

        # We assume that the zeropoints are present in the
        # CCDs file (starting in DR2)
#         z0 = dict(g = 25.08,
#                   r = 25.29,
#                   z = 24.92,)
#         z0 = np.array([z0[f[0]] for f in CCD.filter])
# 
#         good = np.ones(len(CCD), bool)
#         n0 = sum(good)
#         # This is our list of cuts to remove non-photometric CCD images
#         for name,crit in [
#             ('exptime < 30 s', (CCD.exptime < 30)),
#             ('ccdnmatch < 20', (CCD.ccdnmatch < 20)),
#             ('abs(zpt - ccdzpt) > 0.1',
#              (np.abs(CCD.zpt - CCD.ccdzpt) > 0.1)),
#             ('zpt < 0.5 mag of nominal (for DECam)',
#              ((CCD.camera == 'decam') * (CCD.zpt < (z0 - 0.5)))),
#             ('zpt > 0.25 mag of nominal (for DECam)',
#              ((CCD.camera == 'decam') * (CCD.zpt > (z0 + 0.25)))),
#              ]:


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
	arcsec2pix = 1./.262 #from Dustin
	if band == 'g':
		zp0 = 25.08
		recm = 24.
	if band == 'r':
		zp0 = 25.29
		recm = 23.4
	if band == 'z':
		zp0 = 24.92
		recm = 22.5
	nd = 0
	nbr = 0	
	for i in range(0,len(f)):
		if f[i]['filter'] == band:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99:
				if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
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
					if m > recm:
						nbr += 1.
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
	print 'percentage better than requirements '+str(nbr/n)

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title(str(mean)[:5]+r'$\pm$'+str(std)[:4])
	#plt.xscale('log')
	plt.show()
	return True

def plotMaghist2obs(band,ndraw = 1e5,nbin=100):
	#This randomly takes two ccd observations and find the coadded depth base on 1/noise^2tot = 1/(1/noise1^2+1/noise2^2)
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	readnoise = 10. # e-; 7.0 to 15.0 according to DECam Data Handbook
	p = 1.15 #value given in imaging requirements
	gain = 4.0 #from Dustin
	f = fitsio.read(dir+'/legacypipe-dir/decals-ccds.fits.gz')
	nr = len(f)
	NTl = []
	emin = 1000
	emax = 0
	msee = 0
	n = 0
	arcsec2pix = 1./.262 #from Dustin
	if band == 'g':
		zp0 = 25.08
		recm = 24.
	if band == 'r':
		zp0 = 25.29
		recm = 23.4
	if band == 'z':
		zp0 = 24.92
		recm = 22.5
	nd = 0
	nbr = 0	
	#for i in range(0,len(f)):
	while nd < ndraw:
		i = int(random()*nr)
		j = int(random()*nr)
		#j = i #this is used to test result when conditions are exactly the same on each ccd
		if f[i]['filter'] == band and f[j]['filter'] == band:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99:
				if f[j]['seeing'] != 99 and f[j]['ccdzpt'] != 99 and f[j]['fwhm'] != 99:
					if f[j]['dec'] > -20 and f[j]['exptime'] >=30 and f[j]['ccdnmatch'] >= 20 and abs(f[j]['zpt'] - f[j]['ccdzpt']) <= 0.1 and f[j]['zpt'] >= zp0-.5 and f[j]['zpt'] <=zp0+.25:   
						if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
							nd += 1
							avsky = f[i]['avsky']
							skysig = sqrt(avsky * gain + readnoise**2) / gain
							zpscale = zeropointToScale(f[i]['ccdzpt'] + 2.5*log(f[i]['exptime'],10.))
							skysig /= zpscale
							psf_sigma = f[i]['fwhm'] / 2.35
							avsky2 = f[j]['avsky']
							skysig2 = sqrt(avsky2 * gain + readnoise**2) / gain
							zpscale2 = zeropointToScale(f[j]['ccdzpt'] + 2.5*log(f[j]['exptime'],10.))
							skysig2 /= zpscale2
							psf_sigma2 = f[j]['fwhm'] / 2.35
							# point-source depth
							#psfnorm = 1./(2. * sqrt(pi) * psf_sigma) #1/Neff #for point source
							#detsig1 = skysig / psfnorm
							Np = ((4.*pi*psf_sigma**2.)**(1./p) + (8.91*(.45*arcsec2pix)**2. )**(1./p))**p #Neff in requirements doc
							Np2 = ((4.*pi*psf_sigma2**2.)**(1./p) + (8.91*(.45*arcsec2pix)**2. )**(1./p))**p #Neff in requirements doc
							Np = sqrt(Np) #square root necessary because Np gives sum of noise squared
							Np2 = sqrt(Np2)
							detsig1 = skysig*Np #total noise
							detsig2 = skysig2*Np2
							detsigtot = sqrt(1./(1./detsig1**2.+1./detsig2**2))
							m = nanomaggiesToMag(detsigtot * 5.)
							if m > 30 or m < 18:
								print skysig,avsky,f[i]['fwhm'],f[i]['ccdzpt'],f[i]['exptime']
							NTl.append(m)
							if f[i]['exptime'] > emax:
								emax = f[i]['exptime']
							if 	f[i]['exptime'] < emin:
								emin = f[i]['exptime']
							n += 1.
							msee += f[i]['seeing']	
							if m > recm:
								nbr += 1.
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
	print 'percentage better than requirements '+str(nbr/float(nd))

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title('MC 2 exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4])
	#plt.xscale('log')
	plt.show()
	return True

def plotMaghist3obs(band,ndraw = 1e5,nbin=100):
	#This randomly takes three ccd observations and find the coadded depth base on 1/noise^2tot = 1/(1/noise1^2+1/noise2^2)
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	readnoise = 10. # e-; 7.0 to 15.0 according to DECam Data Handbook
	p = 1.15 #value given in imaging requirements
	gain = 4.0 #from Dustin
	f = fitsio.read(dir+'/legacypipe-dir/decals-ccds.fits.gz')
	nr = len(f)
	NTl = []
	emin = 1000
	emax = 0
	msee = 0
	n = 0
	arcsec2pix = 1./.262 #from Dustin
	if band == 'g':
		zp0 = 25.08
		recm = 24.
	if band == 'r':
		zp0 = 25.29
		recm = 23.4
	if band == 'z':
		zp0 = 24.92
		recm = 22.5
	nd = 0	
	#for i in range(0,len(f)):
	nbr = 0
	while nd < ndraw:
		i = int(random()*nr)
		j = int(random()*nr)
		k = int(random()*nr)
		if f[i]['filter'] == band and f[j]['filter'] == band and f[k]['filter'] == band:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99:
				if f[j]['seeing'] != 99 and f[j]['ccdzpt'] != 99 and f[j]['fwhm'] != 99:
					if f[k]['seeing'] != 99 and f[k]['ccdzpt'] != 99 and f[k]['fwhm'] != 99:
						if f[j]['dec'] > -20 and f[j]['exptime'] >=30 and f[j]['ccdnmatch'] >= 20 and abs(f[j]['zpt'] - f[j]['ccdzpt']) <= 0.1 and f[j]['zpt'] >= zp0-.5 and f[j]['zpt'] <=zp0+.25:   
							if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
								if f[k]['dec'] > -20 and f[k]['exptime'] >=30 and f[k]['ccdnmatch'] >= 20 and abs(f[k]['zpt'] - f[k]['ccdzpt']) <= 0.1 and f[k]['zpt'] >= zp0-.5 and f[k]['zpt'] <=zp0+.25:
									nd += 1
									avsky = f[i]['avsky']
									skysig = sqrt(avsky * gain + readnoise**2) / gain
									zpscale = zeropointToScale(f[i]['ccdzpt'] + 2.5*log(f[i]['exptime'],10.))
									skysig /= zpscale
									psf_sigma = f[i]['fwhm'] / 2.35
									avsky2 = f[j]['avsky']
									skysig2 = sqrt(avsky2 * gain + readnoise**2) / gain
									zpscale2 = zeropointToScale(f[j]['ccdzpt'] + 2.5*log(f[j]['exptime'],10.))
									skysig2 /= zpscale2
									psf_sigma2 = f[j]['fwhm'] / 2.35
									avsky3 = f[k]['avsky']
									skysig3 = sqrt(avsky3 * gain + readnoise**2) / gain
									zpscale3 = zeropointToScale(f[k]['ccdzpt'] + 2.5*log(f[k]['exptime'],10.))
									skysig3 /= zpscale3
									psf_sigma3 = f[k]['fwhm'] / 2.35
									# point-source depth
									#psfnorm = 1./(2. * sqrt(pi) * psf_sigma) #1/Neff #for point source
									#detsig1 = skysig / psfnorm
									Np = ((4.*pi*psf_sigma**2.)**(1./p) + (8.91*(.45*arcsec2pix)**2. )**(1./p))**p #Neff in requirements doc
									Np2 = ((4.*pi*psf_sigma2**2.)**(1./p) + (8.91*(.45*arcsec2pix)**2. )**(1./p))**p #Neff in requirements doc
									Np3 = ((4.*pi*psf_sigma3**2.)**(1./p) + (8.91*(.45*arcsec2pix)**2. )**(1./p))**p #Neff in requirements doc
									Np = sqrt(Np) #square root necessary because Np gives sum of noise squared
									Np2 = sqrt(Np2)
									Np3 = sqrt(Np3)
									detsig1 = skysig*Np #total noise
									detsig2 = skysig2*Np2
									detsig3 = skysig3*Np3
									detsigtot = sqrt(1./(1./detsig1**2.+1./detsig2**2+1./detsig3**2))
									m = nanomaggiesToMag(detsigtot * 5.)
									if m > 30 or m < 18:
										print skysig,avsky,f[i]['fwhm'],f[i]['ccdzpt'],f[i]['exptime']
									if m > recm:
										nbr += 1.	
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
	print 'percentage better than requirements '+str(nbr/float(nd))
	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title('MC 3 exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4])
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


		
		