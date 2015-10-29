dir = '$HOME/' # obviously needs to be changed
from math import *

### A couple of useful conversions

def zeropointToScale(zp):
	return 10.**((zp - 22.5)/2.5)	

def nanomaggiesToMag(nm):
	return -2.5 * (log(nm,10.) - 9.)

'''
* CCDNUM = 31 (S7) should mask outside the region [1:1023,1:4094]
That constraint has not yet been included!
'''

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
		DS = f[i]['propid']
		if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
			DS = 1
		if f[i]['filter'] == band:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
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

	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages('DR2DECaLS'+band+'1exposure.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title('1 exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(n))[:5])
	#plt.xscale('log')
	pp.savefig()
	pp.close()
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
		DS = f[i]['propid']
		DS2 = f[j]['propid']
		if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
			DS = 1
		if DS2 == '2014B-0404' or DS2 == '2013A-0741':
			DS2 = 1

		if f[i]['filter'] == band and f[j]['filter'] == band and DS == 1 and DS2 == 1:
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

	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages('DR2DECaLS'+band+'2exposures.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title('MC 2 exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(nd))[:5])
	#plt.xscale('log')
	pp.savefig()
	pp.close()
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
	nl = [] #to speed things up, first make list of noise, then do draws
	for i in range(0,len(f)):
		DS = f[i]['propid']
		if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
			DS = 1
		if f[i]['filter'] == band:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
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
					nl.append(detsig1)
	ng = len(nl)
	print ng
	for nd in range(0,int(ndraw)):
		i = int(random()*ng)
		j = int(random()*ng)
		k = int(random()*ng)
		detsig1 = nl[i]
		detsig2 = nl[j]
		detsig3 = nl[k]
		detsigtot = sqrt(1./(1./detsig1**2.+1./detsig2**2+1./detsig3**2))
		m = nanomaggiesToMag(detsigtot * 5.)
		if m > recm:
			nbr += 1.	
		NTl.append(m)
		n += 1.
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
	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages('DR2DECaLS'+band+'3exposures.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title('MC 3 exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(nd))[:5])
	#plt.xscale('log')
	pp.savefig()
	pp.close()
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


		
		