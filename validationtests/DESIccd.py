dir = '$HOME/' # obviously needs to be changed
localdir = '/Users/ashleyross/DESI/' #place for local DESI stuff
from math import *
from healpix import healpix,radec2thphi
import numpy as np
hpix = healpix()
extmap = np.loadtxt('healSFD_r_256_fullsky.dat')
DECamExt = [3.995,3.214,2.165,1.592,1.211,1.064]

### A couple of useful conversions

def zeropointToScale(zp):
	return 10.**((zp - 22.5)/2.5)	

def nanomaggiesToMag(nm):
	return -2.5 * (log(nm,10.) - 9.)

def Magtonanomaggies(m):
	return 10.**(-m/2.5+9.)
	#-2.5 * (log(nm,10.) - 9.)


'''
* CCDNUM = 31 (S7) should mask outside the region [1:1023,1:4094]
That constraint has not yet been included!
'''

### calculate depth stats in magnitudes

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
		cor = 0.07
		extc = 3.303/2.751
	if band == 'r':
		zp0 = 25.29
		recm = 23.4
		cor = .12
		extc = 2.285/2.751
	if band == 'z':
		zp0 = 24.92
		recm = 22.5
		extc = 1.263/2.751
		cor = .18
	nd = 0
	nbr = 0	
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data
		if f[i]['filter'] == band:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
				if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ra,dec = f[i]['ra'],f[i]['dec']
					th,phi = radec2thphi(ra,dec)
					pix = hpix.ang2pix_nest(256,th,phi)
					ext = extmap[pix]*extc
					#if ext > 0.5:
					#	print ext,extc,ra,dec,pix
						#break
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
					signalext = 1./10.**(-ext/2.5)
					#signalext = 1.
					m = nanomaggiesToMag(detsig1 * 5.*signalext)-cor#-ext
					#if m > 30 or m < 18:
					#	print skysig,avsky,f[i]['fwhm'],f[i]['ccdzpt'],f[i]['exptime']
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
	pp = PdfPages(localdir+'validationplots/DR2DECaLS'+band+'1exposure.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title('1 exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(n))[:5])
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True

def plotMaghist_proc(band,nbin=100):
	#this produces the histogram for Dustin's processed galaxy depth
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	f = fitsio.read(localdir+'decals-ccds-annotated.fits')
	NTl = []
	nd = 0
	nbr = 0	
		
	if band == 'g':
		be = 1
		zp0 = 25.08
		recm = 24.
	if band == 'r':
		be = 2
		zp0 = 25.29
		recm = 23.4
	if band == 'z':
		be = 4
		zp0 = 24.92
		recm = 22.5

	n = 0
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data

		if f[i]['filter'] == band:
			
			if DS == 1:
				
				if f[i]['dec'] > -20 and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True:   
					n += 1
				#if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
				#	if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ext = f[i]['decam_extinction'][be]
					detsig1 = Magtonanomaggies(f[i]['galdepth'])/5. #total noise
					signalext = 1./10.**(-ext/2.5)
				
					m = nanomaggiesToMag(detsig1 * 5.*signalext)
					#m = f[i]['galdepth']-ext
				#if m > 30 or m < 18:
				#	print skysig,avsky,f[i]['fwhm'],f[i]['ccdzpt'],f[i]['exptime']
					if m > 0 and m < 30:
						NTl.append(m)
			
	print n	
	print len(NTl)
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
	pp = PdfPages(localdir+'validationplots/DR2DECaLS'+band+'1exposureprocess.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title('1 exposure processed depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(n))[:5])
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True

def Magcompgauss(band,nbin=100):
	#compare Dustin's Gaussian depths to mine
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	readnoise = 10. # e-; 7.0 to 15.0 according to DECam Data Handbook
	p = 1.15 #value given in imaging requirements
	gain = 4.0 #from Dustin
	f = fitsio.read(localdir+'decals-ccds-annotated.fits')
	NTl = []
	emin = 1000
	emax = 0
	msee = 0
	n = 0
	arcsec2pix = 1./.262 #from Dustin

	if band == 'g':
		zp0 = 25.08
		recm = 24.
		cor = 0.07
		extc = 3.303/2.751
	if band == 'r':
		zp0 = 25.29
		recm = 23.4
		cor = .12
		extc = 2.285/2.751
	if band == 'z':
		zp0 = 24.92
		recm = 22.5
		extc = 1.263/2.751
		cor = .18
	nd = 0
	nbr = 0	
	avem = 0
	avemd = 0
	stdd = 0
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data

		if f[i]['filter'] == band:
			
			if DS == 1:
				
				if f[i]['dec'] > -20 and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True:   
					n += 1
				#if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
				#	if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					#ext = f[i]['decam_extinction'][be]
					avsky = f[i]['avsky']
					skysig = sqrt(avsky * gain + readnoise**2) / gain
					zpscale = zeropointToScale(f[i]['ccdzpt'] + 2.5*log(f[i]['exptime'],10.))
					skysig /= zpscale
					skysig = f[i]['sig1']
					psf_sigma = f[i]['fwhm'] / 2.35
					# point-source depth
					#psfnorm = 1./(2. * sqrt(pi) * psf_sigma) #1/Neff #for point source
					#detsig1 = skysig / psfnorm
					Np = ((4.*pi*psf_sigma**2.)**(1./p) + (8.91*(.45*arcsec2pix)**2. )**(1./p))**p #Neff in requirements doc
					Np = sqrt(Np) #square root necessary because Np gives sum of noise squared
					detsig1 = skysig*Np #total noise
					m = nanomaggiesToMag(detsig1 * 5.)
					detsig1d = Magtonanomaggies(f[i]['gaussgaldepth'])/5. #total noise
					md = nanomaggiesToMag(detsig1d * 5.)

					avem += m
					avemd += md
					stdd += (md-m)**2.
	avem = avem/n
	avemd = avemd/n
	dif = avemd-avem
	stdd = sqrt(stdd/n-dif**2.)
	return avem,avemd,stdd

def Magcompgaussproc(band,nbin=100):
	#compare Dustin's Gaussian depths to the processed ones
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	readnoise = 10. # e-; 7.0 to 15.0 according to DECam Data Handbook
	p = 1.15 #value given in imaging requirements
	gain = 4.0 #from Dustin
	f = fitsio.read(localdir+'decals-ccds-annotated.fits')
	NTl = []
	emin = 1000
	emax = 0
	msee = 0
	n = 0
	arcsec2pix = 1./.262 #from Dustin

	if band == 'g':
		zp0 = 25.08
		recm = 24.
		cor = 0.07
		extc = 3.303/2.751
	if band == 'r':
		zp0 = 25.29
		recm = 23.4
		cor = .12
		extc = 2.285/2.751
	if band == 'z':
		zp0 = 24.92
		recm = 22.5
		extc = 1.263/2.751
		cor = .18
	nd = 0
	nbr = 0	
	avem = 0
	avemd = 0
	stdd = 0
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data

		if f[i]['filter'] == band:
			
			if DS == 1:
				
				if f[i]['dec'] > -20 and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True:   
					n += 1
				#if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
				#	if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					#ext = f[i]['decam_extinction'][be]
					detsig1 = Magtonanomaggies(f[i]['galdepth'])
					m = nanomaggiesToMag(detsig1 )
					detsig1d = Magtonanomaggies(f[i]['gaussgaldepth']) #total noise
					md = nanomaggiesToMag(detsig1d )

					avem += m
					avemd += md
					stdd += (md-m)**2.
	avem = avem/n
	avemd = avemd/n
	dif = avemd-avem
	stdd = sqrt(stdd/n-dif**2.)
	return avem,avemd,stdd


def plotMaghist_proccom(band,nbin=100):
	#this produces the histograms for Dustin's processed galaxy depth and the analytic calculation and compares them
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	f = fitsio.read(localdir+'decals-ccds-annotated.fits')
	readnoise = 10. # e-; 7.0 to 15.0 according to DECam Data Handbook
	p = 1.15 #value given in imaging requirements
	gain = 4.0 #from Dustin
	arcsec2pix = 1./.262 #from Dustin
	
	NTl = []
	NTla = []
	nd = 0
	nbr = 0	
		
	if band == 'g':
		be = 1
		zp0 = 25.08
		recm = 24.
		cor = 0.07+.05
		
	if band == 'r':
		be = 2
		zp0 = 25.29
		recm = 23.4
		cor = .12
	if band == 'z':
		be = 4
		zp0 = 24.92
		recm = 22.5
		cor = .18+.005
		


	n = 0
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data

		if f[i]['filter'] == band:
			
			if DS == 1:
				
				if f[i]['dec'] > -20 and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True:   
					n += 1
				#if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
				#	if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ext = f[i]['decam_extinction'][be]
					detsig1 = Magtonanomaggies(f[i]['galdepth'])/5. #total noise
					signalext = 1./10.**(-ext/2.5)
				
					m = nanomaggiesToMag(detsig1 * 5.*signalext)
					avsky = f[i]['avsky']
					skysig = sqrt(avsky * gain + readnoise**2) / gain
					zpscale = zeropointToScale(f[i]['ccdzpt'] + 2.5*log(f[i]['exptime'],10.))
					skysig /= zpscale
					skysig = f[i]['sig1']
					psf_sigma = f[i]['fwhm'] / 2.35
					# point-source depth
					#psfnorm = 1./(2. * sqrt(pi) * psf_sigma) #1/Neff #for point source
					#detsig1 = skysig / psfnorm
					Np = ((4.*pi*psf_sigma**2.)**(1./p) + (8.91*(.45*arcsec2pix)**2. )**(1./p))**p #Neff in requirements doc
					Np = sqrt(Np) #square root necessary because Np gives sum of noise squared
					detsig1a = skysig*Np #total noise
					#detsig1a = Magtonanomaggies(f[i]['gaussgaldepth'])/5.
					ma = nanomaggiesToMag(detsig1a * 5.*signalext)-cor
					#m = f[i]['galdepth']-ext
				#if m > 30 or m < 18:
				#	print skysig,avsky,f[i]['fwhm'],f[i]['ccdzpt'],f[i]['exptime']
					if m > 0 and m < 30:
						NTl.append(m)
						NTla.append(ma)
			
	print n	
	print len(NTl)
	minN = min(NTl)
	maxN = max(NTl)+.0001
	print minN,maxN
	hl = zeros((nbin))
	for i in range(0,len(NTl)):
		bin = int(nbin*(NTl[i]-minN)/(maxN-minN))
		hl[bin] += 1
	al = zeros((nbin))
	for i in range(0,len(NTla)):
		if NTla[i] > minN and NTla[i] < maxN:
			bin = int(nbin*(NTla[i]-minN)/(maxN-minN))
			al[bin] += 1
	Nl = []
	for i in range(0,len(hl)):
		Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))
	NTl = array(NTl)
	NTla = array(NTla)
	mean = sum(NTl)/float(len(NTl))
	std = sqrt(sum(NTl**2.)/float(len(NTl))-mean**2.)
	meana = sum(NTla)/float(len(NTla))
	stda = sqrt(sum(NTla**2.)/float(len(NTla))-meana**2.)
	stdcomp = 0
	for i in range(0,len(NTl)):
		stdcomp += (NTla[i]-NTl[i])**2.
	stdcomp = sqrt(stdcomp/float(len(NTla)))
	#stdcomp = sqrt(sum(NTla*NTl)/float(len(NTla))-meana*mean)
	NTl.sort()
	NTla.sort()
	if len(NTl)/2. != len(NTl)/2:
		med = NTl[len(NTl)/2+1]
		meda = NTla[len(NTl)/2+1]
	else:
		med = (NTl[len(NTl)/2+1]+NTl[len(NTl)/2])/2.
		meda = (NTla[len(NTla)/2+1]+NTl[len(NTl)/2])/2.
	print mean,meana,med,meda,std,stda,stdcomp
	print 'percentage better than requirements '+str(nbr/n)

	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages(localdir+'validationplots/DR2DECaLS'+band+'1exposureprocesscompa.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.plot(Nl,al,'r-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth',size=16)
	plt.ylabel('# of ccds',size=16)
	if band == 'g':
		plt.xlim(22.5,24)
		plt.text(22.6,1800,'Tractor',color='k',size=16)
		plt.text(22.6,1700,'Gaussian',color='r',size=16)
	#plt.title('1 exposure processed depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(n))[:5])
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True


def plotMaghist_proc_Nexp(band,Nexp,ndraw = 1e5,nbin=100,rel='DR3',survey='decals'):
	#this produces the histogram for Dustin's processed galaxy depth
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	if rel == 'DR2':
		f = fitsio.read(localdir+'decals-ccds-annotated.fits')
	if rel == 'DR3':
		f = fitsio.read(localdir+'ccds-annotated-'+survey+'.fits.gz')	
	nl = []
	nd = 0
	nbr = 0	
		
	if band == 'g':
		be = 1
		zp0 = 25.08
		recm = 24.
	if band == 'r':
		be = 2
		zp0 = 25.29
		recm = 23.4
	if band == 'z':
		be = 4
		zp0 = 24.92
		recm = 22.5

	n = 0
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			#if pid == '2014B-0404' or pid == '2013A-0741':
			DS = 1 #enforce 2015 data

		if f[i]['filter'] == band:
			
			if DS == 1:
				n += 1				
				if f[i]['dec'] > -20 and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True :   
				
				#if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
					#if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ext = f[i]['decam_extinction'][be]
					detsig1 = Magtonanomaggies(f[i]['galdepth'])/5. #total noise
					signalext = 1./10.**(-ext/2.5)
					nl.append(detsig1*signalext)
			
	print n	
	print len(nl)
	ng = len(nl)
	NTl = []
	for nd in range(0,int(ndraw)):
		detsigtoti = 0
		for i in range(0,Nexp):
			ind = int(random()*ng)
			detsig1 = nl[ind]
			detsigtoti += 1./detsig1**2.
		detsigtot = sqrt(1./detsigtoti)
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
	pp = PdfPages(localdir+'validationplots/'+rel+survey+band+str(Nexp)+'exposures.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title('MC '+str(Nexp)+' exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(nd))[:5])
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True

def plotMaghist_proc_sexp(band,ndraw = 1e5,bs=.005,rel='DR3',survey='decals'):
	#this produces the histogram for Dustin's processed galaxy depth, for single exposures
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	if rel == 'DR2':
		f = fitsio.read(localdir+'decals-ccds-annotated.fits')
	if rel == 'DR3':
		f = fitsio.read(localdir+'ccds-annotated-'+survey+'.fits.gz')	
	nl = []
	nd = 0
	nbr = 0	
		
	if band == 'g':
		be = 1
		zp0 = 25.08
		recm = 24.
	if band == 'r':
		be = 2
		zp0 = 25.29
		recm = 23.4
	if band == 'z':
		be = 4
		zp0 = 24.92
		recm = 22.5

	n = 0
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			#if pid == '2014B-0404' or pid == '2013A-0741':
			DS = 1 #enforce 2015 data

		if f[i]['filter'] == band:
			
			if DS == 1:
				n += 1				
				if f[i]['dec'] > -20 and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True :   
				
				#if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
					#if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ext = f[i]['decam_extinction'][be]
					magdepth = f[i]['galdepth']-ext
					#detsig1 = Magtonanomaggies(f[i]['galdepth'])/5. #total noise
					#signalext = 1./10.**(-ext/2.5)
					nl.append(magdepth)
			
	print n	
	print len(nl)
	ng = len(nl)
	minN = min(nl)
	maxN = max(nl)+.0001
	print minN,maxN
	nbin = int((maxN-minN)/bs)
	hl = zeros((nbin))
	for i in range(0,len(nl)):
		bin = int(nbin*(nl[i]-minN)/(maxN-minN))
		hl[bin] += 1
	Nl = []
	for i in range(0,len(hl)):
		Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))
	nl = array(nl)
	mean = sum(nl)/float(len(nl))
	std = sqrt(sum(nl**2.)/float(len(nl))-mean**2.)
	nl.sort()
	if len(nl)/2. != len(nl)/2:
		med = nl[len(nl)/2+1]
	else:
		med = (nl[len(nl)/2+1]+nl[len(nl)/2])/2.
	print mean,med,std
	print 'percentage better than requirements '+str(nbr/float(ng))
	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages(localdir+'validationplots/'+rel+survey+band+'1exposure.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title(rel+' single exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4])#+r', $f_{\rm pass}=$'+str(nbr/float(nd))[:5])
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True


def plotMaghist_proc_sexpDR2DR3(band,ndraw = 1e5,bs=.01,survey='decals'):
	#this produces the normalized histograms for Dustin's processed galaxy depth, for single exposures, comparing DR2 and DR3
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	f2 = fitsio.read(localdir+'decals-ccds-annotated.fits')
	f3 = fitsio.read(localdir+'ccds-annotated-'+survey+'.fits.gz')	
	nl = []
	nd = 0
	nbr = 0	
		
	if band == 'g':
		be = 1
		zp0 = 25.08
		recm = 24.
	if band == 'r':
		be = 2
		zp0 = 25.29
		recm = 23.4
	if band == 'z':
		be = 4
		zp0 = 24.92
		recm = 22.5

	n = 0
	for i in range(0,len(f3)):
		pid = f3[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f3[i]['date_obs'].split('-')[0])
		if year > 2014:
			#if pid == '2014B-0404' or pid == '2013A-0741':
			DS = 1 #enforce 2015 data

		if f3[i]['filter'] == band:
			
			if DS == 1:
				n += 1				
				if f3[i]['dec'] > -20 and f3[i]['photometric'] == True and f3[i]['blacklist_ok'] == True and f3[i]['tileid'] > 0 :   
				
				#if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
					#if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ext = f3[i]['decam_extinction'][be]
					magdepth = f3[i]['galdepth']-ext
					#detsig1 = Magtonanomaggies(f[i]['galdepth'])/5. #total noise
					#signalext = 1./10.**(-ext/2.5)
					nl.append(magdepth)

	nl2 = []
	for i in range(0,len(f2)):
		pid = f2[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f2[i]['date_obs'].split('-')[0])
		if year > 2014:
			#if pid == '2014B-0404' or pid == '2013A-0741':
			DS = 1 #enforce 2015 data

		if f2[i]['filter'] == band:
			
			if DS == 1:
				n += 1				
				if f2[i]['dec'] > -20 and f2[i]['photometric'] == True and f2[i]['blacklist_ok'] == True and f2[i]['tileid'] > 0:   
				
				#if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
					#if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ext = f2[i]['decam_extinction'][be]
					magdepth = f2[i]['galdepth']-ext
					#detsig1 = Magtonanomaggies(f[i]['galdepth'])/5. #total noise
					#signalext = 1./10.**(-ext/2.5)
					nl2.append(magdepth)
			
	print n	
	print len(nl),len(nl2)
	ng = len(nl)
	minN = min(nl)
	maxN = max(nl)+.0001
	print minN,maxN
	nbin = int((maxN-minN)/bs)
	hl = zeros((nbin))
	for i in range(0,len(nl)):
		bin = int(nbin*(nl[i]-minN)/(maxN-minN))
		hl[bin] += 1
	Nl = []
	for i in range(0,len(hl)):
		Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))
	ng2 = len(nl2)
	minN2 = min(nl2)
	maxN2 = max(nl2)+.0001
	print minN2,maxN2
	nbin2 = int((maxN2-minN2)/bs)
	hl2 = zeros((nbin))
	for i in range(0,len(nl2)):
		bin = int(nbin2*(nl2[i]-minN2)/(maxN2-minN2))
		hl2[bin] += 1
	Nl2 = []
	for i in range(0,len(hl2)):
		Nl2.append(minN2+i*(maxN2-minN2)/float(nbin2)+0.5*(maxN2-minN2)/float(nbin2))

	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages(localdir+'validationplots/DR2DR3'+survey+band+'1exposure.pdf')	

	plt.plot(Nl,hl/float(len(nl)),'k-')
	plt.plot(Nl2,hl2/float(len(nl2)),'r-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('Normalized # of ccds')
	plt.title('single exposure depth, comparing DR2 and DR3')
	plt.text(minN+.5,.9*(max(hl2))/float(len(nl2)),'DR3',color='k')
	plt.text(minN+.5,.85*(max(hl2))/float(len(nl2)),'DR2',color='r')
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True


def plotMaghist_proc_Nexpcompdata(band,Nexp,ndraw = 1e5,nbin=100):
	#this produces the histogram for Dustin's processed galaxy depth
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	f = fitsio.read(localdir+'decals-ccds-annotated.fits')
	nl = []
	nd = 0
	nbr = 0	
		
	if band == 'g':
		be = 1
		zp0 = 25.08
		recm = 24.
		extc = 3.303
	if band == 'r':
		be = 2
		zp0 = 25.29
		recm = 23.4
		extc = 2.285
	if band == 'z':
		be = 4
		zp0 = 24.92
		recm = 22.5
		extc = 1.263
	n = 0
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data

		if f[i]['filter'] == band:
			
			if DS == 1:
				n += 1
				#if f[i]['dec'] > -20 #and f[i]['photometric'] == False and f[i]['blacklist_ok'] == False :   
				if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
					if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
						ext = f[i]['decam_extinction'][be]
						detsig1 = Magtonanomaggies(f[i]['galdepth'])/5. #total noise
						signalext = 1./10.**(-ext/2.5)
						nl.append(detsig1*signalext)
			
	print n	
	print len(nl)
	ng = len(nl)
	NTl = []
	for nd in range(0,int(ndraw)):
		detsigtoti = 0
		for i in range(0,Nexp):
			ind = int(random()*ng)
			detsig1 = nl[ind]
			detsigtoti += 1./detsig1**2.
		detsigtot = sqrt(1./detsigtoti)
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
	print 'percentage better than requirement '+str(nbr/float(nd))
	f = fitsio.read(localdir+'dr2_v1.0_gal_nsig5_'+band+'_depth_ebv.fit')
	dl = zeros((nbin))
	ndt = 0
	for i in range(0,len(f)):
		if f[i]['NEXP'] > Nexp-.1 and f[i]['NEXP'] < Nexp + .1:
			depth = f[i]['DEPTH']-extc*f[i]['EBV']
			if depth > minN and depth < maxN:
				bin = int(nbin*(depth-minN)/(maxN-minN))
				dl[bin] += 1.
				ndt += 1.
	sd = 0
	for i in range(0,len(dl)):
		sd += dl[i]
		if sd/ndt > .33:
			per = Nl[i]
			break
	print '67th percential is '+str(per)		
	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages(localdir+'validationplots/DR2DECaLSMCcompd'+band+str(Nexp)+'exposures.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.plot(Nl,dl*max(hl)/max(dl),'r-')
	plt.text(minN+.05*(maxN-minN),.9*max(hl),'MC result from ccd file',color='k')
	plt.text(minN+.05*(maxN-minN),.825*max(hl),r'From maps ($\pm$ 0.1 exposure)',color='r')
	
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('relative fraction (arbitrary)')
	plt.title('MC '+str(Nexp)+' exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(nd))[:5])
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True

def plotMaghist_surveydata(band,nbin=100):
	#this produces the histogram for Dustin's processed galaxy depth
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	nl = []
	nd = 0
	nbr = 0	
		
	if band == 'g':
		be = 1
		zp0 = 25.08
		recm = 24.
		extc = 3.303
	if band == 'r':
		be = 2
		zp0 = 25.29
		recm = 23.4
		extc = 2.285
	if band == 'z':
		be = 4
		zp0 = 24.92
		recm = 22.5
		extc = 1.263
	n = 0
	ntl = [0,0,0,0,0]
	dall = [[],[],[],[],[]]
	f = fitsio.read(localdir+'dr2_v1.0_gal_nsig5_'+band+'_depth_ebv.fit')
	dl = zeros((nbin))
	ndt = 0
	maxN = 0
	minN = 1000
	for i in range(0,len(f)):
		Nexp = int(f[i]['NEXP']/.89)
		if Nexp > 5:
			Nexp = 5
		#for Nexp in range(1,6):
			#if f[i]['NEXP'] > Nexp-.1 and f[i]['NEXP'] < Nexp + .1:
		depth = f[i]['DEPTH']-extc*f[i]['EBV']
		dall[Nexp-1].append(depth)
		ntl[Nexp-1] += 1.
		if depth < minN:
			minN = depth
		if depth > maxN:
			maxN = depth
	ntot = sum(ntl)
	fracl = array(ntl)/ntot
	print fracl
	fracexpl = [0.02,0.24,0.5,0.22,0.02]
	facl =  array(fracexpl)/fracl
	print facl
	for i in range(0,5):
		for j in range(0,len(dall[i])):
			depth = dall[i][j]
			if depth > minN and depth < maxN:
				bin = int(nbin*(depth-minN)/(maxN-minN))
				dl[bin] += 1.*facl[i]
	Nl = []
	for i in range(0,len(dl)):
		Nl.append(minN+i*(maxN-minN)/float(nbin)+0.5*(maxN-minN)/float(nbin))

	sd = 0
	for i in range(0,len(dl)):
		sd += dl[i]/sum(dl)
		if sd > .1:
			per = Nl[i]
			break
	print '90th percential is '+str(per)		
	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages(localdir+'validationplots/DR2DECaLSdatasimfullsurvey_'+band+'.pdf')	

	plt.text(minN+.05*(maxN-minN),.9*max(dl)/sum(dl),'90% depth '+str(per)[:5],color='k')
	plt.plot(Nl,dl/sum(dl),'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('relative fraction (arbitrary)')
	plt.title('Observed data, adjusted for final assumed exposure coverage')
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True


def plotMaghist_proc_Survey(band,ndraw = 1e5,nbin=100):
	#this produces the histogram for Dustin's processed galaxy depth
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	Ftiles =  [0.02,0.26,0.76,0.98]
	f = fitsio.read(localdir+'decals-ccds-annotated.fits')
	nl = []
	nd = 0
	nbr = 0	
		
	if band == 'g':
		be = 1
		zp0 = 25.08
		recm = 24.
	if band == 'r':
		be = 2
		zp0 = 25.29
		recm = 23.4
	if band == 'z':
		be = 4
		zp0 = 24.92
		recm = 22.5

	n = 0
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data

		if f[i]['filter'] == band:
			
			if DS == 1:
				n += 1
				if f[i]['dec'] > -20 and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True:   
				#if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
					#if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ext = f[i]['decam_extinction'][be]
					detsig1 = Magtonanomaggies(f[i]['galdepth'])/5. #total noise
					signalext = 1./10.**(-ext/2.5)
					nl.append(detsig1*signalext)
			
	print n	
	print len(nl)
	ng = len(nl)
	NTl = []
	nbr3 = 0
	nbr6 = 0
	for nd in range(0,int(ndraw)):
		detsigtoti = 0
		ran = random()
		if ran < Ftiles[0]:
			nexp = 1
		if ran > Ftiles[0] and ran < Ftiles[1]:
			nexp = 2
		if ran > Ftiles[1] and ran < Ftiles[2]:
			nexp = 3
		if ran > Ftiles[2] and ran < Ftiles[3]:
			nexp = 4
		if ran > Ftiles[3]:
			nexp = 5	

		for i in range(0,nexp):
			ind = int(random()*ng)
			detsig1 = nl[ind]
			detsigtoti += 1./detsig1**2.
		detsigtot = sqrt(1./detsigtoti)
		m = nanomaggiesToMag(detsigtot * 5.)
		if m > recm:
			nbr += 1.	
		if m > recm-.3:
			nbr3 += 1.
		if m > recm-.6:
			nbr6 += 1.	
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
	nc = 0
	for i in range(0,len(hl)):
		nc += hl[i]
		if nc > .1*nd:
			m9 = Nl[i]
			break
	NTl.sort()
	if len(NTl)/2. != len(NTl)/2:
		med = NTl[len(NTl)/2+1]
	else:
		med = (NTl[len(NTl)/2+1]+NTl[len(NTl)/2])/2.
	print mean,med,std
	print 'percentage better than requirements '+str(nbr/float(nd))
	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages(localdir+'validationplots/DR2DECaLS'+band+'_simsurvey_proc.pdf')	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(Nl,hl,'k-')
	ax.set_xlabel(r'5$\sigma$ '+band+ ' depth',size=16)
	ax.set_ylabel('forecasted relative amoutn of area',size=16)
	#ax.set_title('MC survey depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{r}=$'+str(nbr/float(nd))[:5]+r', $f_{r-0.3}=$'+str(nbr3/float(nd))[:5]+r', $f_{r-0.6}=$'+str(nbr6/float(nd))[:5])
	#ax.text(.4,.9,r'90% depth '+str(m9)[:5], verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,fontsize=15)
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True


def plotMagMap(band,size=4):
	import fitsio
	from matplotlib import pyplot as plt
	import matplotlib.cm as cm
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
		cor = 0.08
		extc = 3.303/2.751
	if band == 'r':
		zp0 = 25.29
		recm = 23.4
		cor = .16
		extc = 2.285/2.751
	if band == 'z':
		zp0 = 24.92
		recm = 22.5
		extc = 1.263/2.751
		cor = .29
	nd = 0
	nbr = 0	
	ral = []
	decl = []
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data
		if f[i]['filter'] == band:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
				if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ra,dec = f[i]['ra'],f[i]['dec']
					th,phi = radec2thphi(ra,dec)
					pix = hpix.ang2pix_nest(256,th,phi)
					ext = extmap[pix]*extc
					#if ext > 0.5:
					#	print ext,extc,ra,dec,pix
						#break
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
					m = nanomaggiesToMag(detsig1 * 5.)-cor-ext
					#if m > 30 or m < 18:
					#	print skysig,avsky,f[i]['fwhm'],f[i]['ccdzpt'],f[i]['exptime']
					NTl.append(m)
					ral.append(ra)
					decl.append(dec)
					if f[i]['exptime'] > emax:
						emax = f[i]['exptime']
					if 	f[i]['exptime'] < emin:
						emin = f[i]['exptime']
					n += 1.
					msee += f[i]['seeing']	
					if m > recm:
						nbr += 1.

	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages(localdir+'validationplots/DR2DECaLS'+band+'map.pdf')	

	#col = (NTl-min(NTl))/(max(NTl)-min(NTl))
	col = NTl
	map = plt.scatter(ral,decl,c=col,s=size,cmap=cm.rainbow,lw=0)
	cbar = plt.colorbar(map)
	cbar.set_label('depth', rotation=270)
	plt.xlabel('r.a. (degrees)')
	plt.ylabel('declination (degrees)')
	plt.title('Map of 1 exposure depth for DR2 '+band+'-band')
	#plt.show()
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
		cor = 0.08
		extc = 3.303/2.751
	if band == 'r':
		zp0 = 25.29
		recm = 23.4
		cor = .16
		extc = 2.285/2.751
	if band == 'z':
		zp0 = 24.92
		recm = 22.5
		extc = 1.263/2.751
		cor = .29
	nd = 0
	nbr = 0	
	#for i in range(0,len(f)):
	while nd < ndraw:
		i = int(random()*nr)
		j = int(random()*nr)
		#j = i #this is used to test result when conditions are exactly the same on each ccd
		pid2 = f[j]['propid']
		pid = f[i]['propid']
		DS = 0
		DS2 = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data
		year2 = int(f[j]['date_obs'].split('-')[0])
		if year2 > 2014:
			if pid2 == '2014B-0404' or pid2 == '2013A-0741':
				DS2 = 1 #enforce 2015 data

		if f[i]['filter'] == band and f[j]['filter'] == band and DS == 1 and DS2 == 1:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99:
				if f[j]['seeing'] != 99 and f[j]['ccdzpt'] != 99 and f[j]['fwhm'] != 99:
					if f[j]['dec'] > -20 and f[j]['exptime'] >=30 and f[j]['ccdnmatch'] >= 20 and abs(f[j]['zpt'] - f[j]['ccdzpt']) <= 0.1 and f[j]['zpt'] >= zp0-.5 and f[j]['zpt'] <=zp0+.25:   
						if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
							nd += 1
							ra,dec = f[i]['ra'],f[i]['dec']
							th,phi = radec2thphi(ra,dec)
							pix = hpix.ang2pix_nest(256,th,phi)
							ext = extmap[pix]*extc
							ra2,dec2 = f[j]['ra'],f[j]['dec']
							th,phi = radec2thphi(ra2,dec2)
							pix2 = hpix.ang2pix_nest(256,th,phi)
							ext2 = extmap[pix2]*extc

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
							signalext = 1./10.**(-ext/2.5)
							signalext2 = 1./10.**(-ext2/2.5)
							detsig1 = skysig*Np #total noise
							detsig2 = skysig2*Np2
							detsigtot = sqrt(1./(1./(detsig1*signalext)**2.+1./(detsig2*signalext2)**2))
							m = nanomaggiesToMag(detsigtot * 5.)-cor#-(ext+ext2)/2.
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
	pp = PdfPages(localdir+'validationplots/DR2DECaLS'+band+'2exposures.pdf')	

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
		cor = 0.08
		extc = 3.303/2.751
	if band == 'r':
		zp0 = 25.29
		recm = 23.4
		cor = .16
		extc = 2.285/2.751
	if band == 'z':
		zp0 = 24.92
		recm = 22.5
		extc = 1.263/2.751
		cor = .29
	nd = 0	
	#for i in range(0,len(f)):
	nbr = 0
	nl = [] #to speed things up, first make list of noise, then do draws
	extl = [] #same for extinction
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data
		if f[i]['filter'] == band:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
				if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ra,dec = f[i]['ra'],f[i]['dec']
					th,phi = radec2thphi(ra,dec)
					pix = hpix.ang2pix_nest(256,th,phi)
					ext = extmap[pix]*extc
					extl.append(ext)
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
					signalext = 1./10.**(-ext/2.5)
							

					detsig1 = skysig*Np #total noise
					nl.append(detsig1*signalext)
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
		m = nanomaggiesToMag(detsigtot * 5.)-cor#-(extl[i]+extl[j]+extl[k])/3.
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
	pp = PdfPages(localdir+'validationplots/DR2DECaLS'+band+'3exposures.pdf')	

	plt.plot(Nl,hl,'k-')
	plt.xlabel(r'5$\sigma$ '+band+ ' depth')
	plt.ylabel('# of ccds')
	plt.title('MC 3 exposure depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{\rm pass}=$'+str(nbr/float(nd))[:5])
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True

def plotMaghist_survey(band,ndraw = 1e5,nbin=100,magmin=0):
	#This takes random selections of 1,2,3,4 and 5 exposures in a fraction matching the expected coverage
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	Ftiles =  [0.02,0.26,0.76,0.98] #cumulative fraction of area
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
		cor = 0.08
		extc = 3.303/2.751
	if band == 'r':
		zp0 = 25.29
		recm = 23.4
		cor = .16
		extc = 2.285/2.751
	if band == 'z':
		zp0 = 24.92
		recm = 22.5
		extc = 1.263/2.751
		cor = .29
	nd = 0	
	#for i in range(0,len(f)):
	nbr = 0
	nl = [] #to speed things up, first make list of noise, then do draws
	extl = [] #same for extinction
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year > 2014:
			if pid == '2014B-0404' or pid == '2013A-0741':
				DS = 1 #enforce 2015 data
		if f[i]['filter'] == band:
			if f[i]['seeing'] != 99 and f[i]['ccdzpt'] != 99 and f[i]['fwhm'] != 99 and DS == 1:
				if f[i]['dec'] > -20 and f[i]['exptime'] >=30 and f[i]['ccdnmatch'] >= 20 and abs(f[i]['zpt'] - f[i]['ccdzpt']) <= 0.1 and f[i]['zpt'] >= zp0-.5 and f[i]['zpt'] <=zp0+.25:   
					ra,dec = f[i]['ra'],f[i]['dec']
					th,phi = radec2thphi(ra,dec)
					pix = hpix.ang2pix_nest(256,th,phi)
					ext = extmap[pix]*extc
					extl.append(ext)
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
					signalext = 1./10.**(-ext/2.5)
					m = nanomaggiesToMag(detsig1 * 5.)-cor-ext
					if m > magmin:
						nl.append(detsig1*signalext)
	ng = len(nl)
	print ng
	nbr3 = 0
	nbr6 = 0
	for nd in range(0,int(ndraw)):
		ran = random()
		if ran < Ftiles[0]:
			nexp = 1
		if ran > Ftiles[0] and ran < Ftiles[1]:
			nexp = 2
		if ran > Ftiles[1] and ran < Ftiles[2]:
			nexp = 3
		if ran > Ftiles[2] and ran < Ftiles[3]:
			nexp = 4
		if ran > Ftiles[3]:
			nexp = 5	
		detsigtoti = 0
		extsum = 0
		for j in range(0,nexp):
			k = int(random()*ng)
			detsigtoti += 1./nl[k]**2.
			extsum += extl[k]
		extcorr = extsum/float(nexp)
		detsigtot = sqrt(1./detsigtoti)	
		m = nanomaggiesToMag(detsigtot * 5.)-cor#-extcorr
		if m > recm:
			nbr += 1.	
		if m > recm-.3:
			nbr3 += 1.
		if m > recm-.6:
			nbr6 += 1.	
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
	nc = 0
	for i in range(0,len(hl)):
		nc += hl[i]
		if nc > .1*nd:
			m9 = Nl[i]
			break
	NTl.sort()
	if len(NTl)/2. != len(NTl)/2:
		med = NTl[len(NTl)/2+1]
	else:
		med = (NTl[len(NTl)/2+1]+NTl[len(NTl)/2])/2.
	print mean,med,std
	print 'percentage better than requirements '+str(nbr/float(nd))
	from matplotlib.backends.backend_pdf import PdfPages
	plt.clf()
	pp = PdfPages(localdir+'validationplots/DR2DECaLS'+band+'_simsurvey.pdf')	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(Nl,hl,'k-')
	ax.set_xlabel(r'5$\sigma$ '+band+ ' depth')
	ax.set_ylabel('# of ccds')
	ax.set_title('MC survey depth '+str(mean)[:5]+r'$\pm$'+str(std)[:4]+r', $f_{r}=$'+str(nbr/float(nd))[:5]+r', $f_{r-0.3}=$'+str(nbr3/float(nd))[:5]+r', $f_{r-0.6}=$'+str(nbr6/float(nd))[:5])
	ax.text(.4,.9,r'90% depth '+str(m9)[:5], verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,fontsize=15)
	#plt.xscale('log')
	pp.savefig()
	pp.close()
	return True

def plotradecccdfile(band,rel='DR3',survey='decals',decmin=-20,yearmin=2015):
	#simply plot ra,dec of all good ccds
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	from random import random
	if rel == 'DR2':
		f = fitsio.read(localdir+'decals-ccds-annotated.fits')
	if rel == 'DR3':
		f = fitsio.read(localdir+'ccds-annotated-'+survey+'.fits.gz')	
	ral = []
	decl = []
	n = 0
	for i in range(0,len(f)):
		pid = f[i]['propid']
		#if DS == '2014B-0404' or DS == '2013A-0741': #enforce DECaLS only
		#	DS = 1
		DS = 0
		year = int(f[i]['date_obs'].split('-')[0])
		if year >= yearmin:
			#if pid == '2014B-0404' or pid == '2013A-0741':
			DS = 1 #enforce 2015 data

		if f[i]['filter'] == band:
			
			if DS == 1:
				n += 1				
				if f[i]['dec'] > decmin and f[i]['photometric'] == True and f[i]['blacklist_ok'] == True: 
					ral.append(f[i]['ra1'])
					decl.append(f[i]['dec1'])
	plt.plot(ral,decl,'k,')
	plt.show()
	return True
  
def plotBorismap(band,prop,op='mean',survey='DECaLS_DR3',nside='1024',oversamp='1'):
	import fitsio
	from matplotlib import pyplot as plt
	import matplotlib.cm as cm
	from numpy import zeros,array
	from healpix import pix2ang_ring,thphi2radec
	import healpy as hp
	f = fitsio.read(localdir+survey+'/nside'+nside+'_oversamp'+oversamp+'/'+survey+'_band_'+band+'_nside'+nside+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz')
	ral = []
	decl = []
	val = f['SIGNAL']
	print min(val),max(val),len(f)/(float(nside)**2.*12)*360*360./pi
	for i in range(0,len(f)):
		#th,phi = pix2ang_ring(4096,f[i]['PIXEL'])
		th,phi = hp.pix2ang(int(nside),f[i]['PIXEL'])
		ra,dec = thphi2radec(th,phi)
		ral.append(ra)
		decl.append(dec)
	print min(val),max(val)	
	map = plt.scatter(ral,decl,c=val,cmap=cm.rainbow,lw=0)
	cbar = plt.colorbar(map)
	cbar.set_label(prop, rotation=270)
	plt.xlabel('r.a. (degrees)')
	plt.ylabel('declination (degrees)')
	plt.title('Map of '+prop +' for '+survey+' '+band+'-band')
	plt.show()
	#plt.xscale('log')
	#pp.savefig()
	#pp.close()
	return True

def plotdepthfromIvar(band,depthmin=23.8,mjdmax='',prop='ivar',op='total',survey='DECaLS_DR3',nside='1024',oversamp='1'):
	import fitsio
	from matplotlib import pyplot as plt
	import matplotlib.cm as cm
	from numpy import zeros,array
	import healpix
	
	from healpix import pix2ang_ring,thphi2radec
	h = healpix.healpix()
	import healpy as hp
	f = fitsio.read(localdir+survey+mjdmax+'/nside'+nside+'_oversamp'+oversamp+'/'+survey+mjdmax+'_band_'+band+'_nside'+nside+'_oversamp'+oversamp+'_'+prop+'__'+op+'.fits.gz')
	ral = []
	decl = []
	val = f['SIGNAL']
	magval = []
	for i in range(0,len(val)):
		magval.append(nanomaggiesToMag(sqrt(1./val[i]) * 5.))
	print min(magval),max(magval),len(f)/(float(nside)**2.*12)*360*360./pi
	if band == 'g':
		extc = 3.303/2.751
	if band == 'r':
		extc = 2.285/2.751
	if band == 'z':
		extc = 1.263/2.751
	magvalgmin = []
	for i in range(0,len(f)):
		#th,phi = pix2ang_ring(4096,f[i]['PIXEL'])
		th,phi = hp.pix2ang(int(nside),f[i]['PIXEL'])
		ra,dec = thphi2radec(th,phi)
		pix256 = h.ang2pix_nest(256,th,phi)
		magval[i] -= extmap[pix256]
		if magval[i] > depthmin:
			ral.append(ra)
			decl.append(dec)
			magvalgmin.append(magval[i])
	print len(ral)/(float(nside)**2.*12)*360*360./pi		
	#print min(val),max(val)	
	map = plt.scatter(ral,decl,c=magvalgmin,cmap=cm.rainbow,lw=0,s=.1)
	cbar = plt.colorbar(map)
	cbar.set_label(r'5$\sigma$ galaxy depth', rotation=270)
	plt.xlabel('r.a. (degrees)')
	plt.ylabel('declination (degrees)')
	plt.title('Map of depth' +' for '+survey+' '+band+'-band')
	plt.show()
	#plt.xscale('log')
	#pp.savefig()
	#pp.close()
	return True
		
		

###Test DR2 region


def Magcomp(band):
	import fitsio
	from matplotlib import pyplot as plt
	from numpy import zeros,array
	readnoise = 10. # e-; 7.0 to 15.0 according to DECam Data Handbook
	p = 1.15 #value given in imaging requirements
	gain = 4.0 #from Dustin
	f = fitsio.read(localdir+'testregion/decals-2444p120-ccds.fits')
	dpm = fitsio.read(localdir+'testregion/decals-2444p120-galdepth-'+band+'.fits')
	expm = fitsio.read(localdir+'testregion/decals-2444p120-nexp-'+band+'.fits')
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
		dpl = []	
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
					minx = f[i]['brick_x0']
					if minx < 0:
						minx = 0
					maxx = f[i]['brick_x1']
					if maxx > 3600:
						maxx = 3600
					miny = f[i]['brick_y0']
					if miny < 0:
						miny = 0
					maxy = f[i]['brick_y1']
					if maxy > 3600:
						maxy = 3600
					for x in range(minx,maxx):
						for y in range(miny,maxy):
							nexp = expm[x][y]
							if nexp == 1:								
								mm = nanomaggiesToMag(1./sqrt(dpm[x][y]) * 5.)
								dpl.append(mm)
		print len(dpl)
		if len(dpl) > 0:
			print m,sum(dpl)/float(len(dpl))
			NTl.append((m,sum(dpl)/float(len(dpl))))
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


		
		