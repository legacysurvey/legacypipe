import astropy.io.fits as fits
import astropy.cosmology as co
c1 = co.Planck15
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import scipy.stats as st

from legacyanalysis.pathnames import get_indir,get_outdir,make_dir

f = open(os.path.join(get_outdir('dr23'),"out-ELG-decals.txt"),'w')

f.write("field 242 \n")

cFile1 = os.path.join(get_indir('dr23'),"catalog-EDR-242.2-243.7-8-12.1-DR3.fits")
f.write("total   primary  any mask \n")

hdu = fits.open(cFile1)
dat = hdu[1].data
prim = (dat['brick_primary'])
noJunk = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
psf = (noJunk) & (dat['type'] == "PSF")
areaC1 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )
f.write("DR3 : "+str(len(dat))+" "+str( len(prim.nonzero()[0]))+" "+str( len(noJunk.nonzero()[0]))+" "+str( len(psf.nonzero()[0]))+" "+str( areaC1)+"\n")
f.write("DR3 : "+str(numpy.round(len(dat)/areaC1))+" "+str( numpy.round(len(prim.nonzero()[0])/areaC1))+" "+str( numpy.round(len(noJunk.nonzero()[0]) /areaC1) )+" "+str( numpy.round(len(psf.nonzero()[0]) /areaC1) )+"\n")

cFile1 =os.path.join(get_indir('dr23'), "catalog-EDR-242.2-243.7-8-12.1-DR2.fits")

hdu = fits.open(cFile1)
dat = hdu[1].data
prim = (dat['brick_primary'])
noJunk = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
psf = (noJunk) & (dat['type'] == "PSF")
areaC1 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )
f.write("DR2 : "+str(len(dat))+" "+str( len(prim.nonzero()[0]))+" "+str( len(noJunk.nonzero()[0]))+" "+str( len(psf.nonzero()[0]))+" "+str( areaC1)+"\n")
f.write("DR2 : "+str(numpy.round(len(dat)/areaC1))+" "+str( numpy.round(len(prim.nonzero()[0])/areaC1))+" "+str( numpy.round(len(noJunk.nonzero()[0]) /areaC1) )+" "+str( numpy.round(len(psf.nonzero()[0]) /areaC1) )+"\n")

f.write("field 351 \n")

cFile2 = os.path.join(get_indir('dr23'),"catalog-EDR-351.25-353.75-m0.1-0.6-DR3.fits")

hdu = fits.open(cFile2)
dat = hdu[1].data
prim = (dat['brick_primary'])
noJunk = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
psf = (noJunk) & (dat['type'] == "PSF")
areaC2 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )
f.write("DR3 : "+str(len(dat))+" "+str( len(prim.nonzero()[0]))+" "+str( len(noJunk.nonzero()[0]))+" "+str( len(psf.nonzero()[0]))+" "+str( areaC2)+"\n")
f.write("DR3 : "+str(numpy.round(len(dat)/areaC2))+" "+str( numpy.round(len(prim.nonzero()[0])/areaC2))+" "+str( numpy.round(len(noJunk.nonzero()[0]) /areaC2) )+" "+str( numpy.round(len(psf.nonzero()[0]) /areaC2) )+"\n")

cFile2 = os.path.join(get_indir('dr23'),"catalog-EDR-351.25-353.75-m0.1-0.6-DR2.fits")

hdu = fits.open(cFile2)
dat = hdu[1].data
prim = (dat['brick_primary'])
noJunk = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
psf = (noJunk) & (dat['type'] == "PSF")

areaC2 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )

f.write("DR2 : "+str(len(dat))+" "+str( len(prim.nonzero()[0]))+" "+str( len(noJunk.nonzero()[0]))+" "+str( len(psf.nonzero()[0]))+" "+str( areaC2)+"\n")
f.write("DR2 : "+str(numpy.round(len(dat)/areaC2))+" "+str( numpy.round(len(prim.nonzero()[0])/areaC2))+" "+str( numpy.round(len(noJunk.nonzero()[0]) /areaC2) )+" "+str( numpy.round(len(psf.nonzero()[0]) /areaC2) )+"\n")

####################################
# NOW HISTOGRAMS
####################################
binG = numpy.arange(10, 28.6, 0.22)
binGSNR = numpy.logspace(-1, 3, 100 )

cFile3 = os.path.join(get_indir('dr23'),"catalog-EDR-351.25-353.75-m0.1-0.6-DR3.fits")
cFile2 = os.path.join(get_indir('dr23'),"catalog-EDR-351.25-353.75-m0.1-0.6-DR2.fits")

hdu = fits.open(cFile3)
dat = hdu[1].data
noJunk3 = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
dr3 = dat[noJunk3]
areaDR3 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )

hdu = fits.open(cFile2)
dat = hdu[1].data
noJunk2 = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
dr2 = dat[noJunk2]
areaDR2 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )

g_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2['decam_flux'].T[1] / dr2['decam_mw_transmission'].T[1])
r_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2['decam_flux'].T[2] / dr2['decam_mw_transmission'].T[2])
z_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2['decam_flux'].T[4] / dr2['decam_mw_transmission'].T[4])
g_e_dr2 = dr2['decam_flux'].T[1] * dr2['decam_flux_ivar'].T[1]**(0.5) 
r_e_dr2 = dr2['decam_flux'].T[2] * dr2['decam_flux_ivar'].T[2]**(0.5) 
z_e_dr2 = dr2['decam_flux'].T[4] * dr2['decam_flux_ivar'].T[4]**(0.5) 

g_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3['decam_flux'].T[1] / dr3['decam_mw_transmission'].T[1])
r_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3['decam_flux'].T[2] / dr3['decam_mw_transmission'].T[2])
z_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3['decam_flux'].T[4] / dr3['decam_mw_transmission'].T[4])
g_e_dr3 = dr3['decam_flux'].T[1] * dr3['decam_flux_ivar'].T[1]**(0.5) 
r_e_dr3 = dr3['decam_flux'].T[2] * dr3['decam_flux_ivar'].T[2]**(0.5) 
z_e_dr3 = dr3['decam_flux'].T[4] * dr3['decam_flux_ivar'].T[4]**(0.5) 

gok = (g_mag_dr2>0)&(g_mag_dr2!=numpy.inf)
rok = (r_mag_dr2>0)&(r_mag_dr2!=numpy.inf)
zok = (z_mag_dr2>0)&(z_mag_dr2!=numpy.inf)
f.write("mags : g ok    r ok   z ok \n")
f.write("DR2 : " + str(len(gok.nonzero()[0])) + " "+str(len(rok.nonzero()[0])) +" "+ str(len(zok.nonzero()[0])) +"\n")
f.write("DR2 : " + str(numpy.round(len(gok.nonzero()[0])/ areaDR2)) +" "+ str(numpy.round(len(rok.nonzero()[0])/ areaDR2)) + " "+str(numpy.round(len(zok.nonzero()[0])/ areaDR2)) +"\n")
gok = (g_mag_dr3>0)&(g_mag_dr3!=numpy.inf)
rok = (r_mag_dr3>0)&(r_mag_dr3!=numpy.inf)
zok = (z_mag_dr3>0)&(z_mag_dr3!=numpy.inf)
f.write("DR3 : " + str(len(gok.nonzero()[0])) + " "+str(len(rok.nonzero()[0])) + " "+str(len(zok.nonzero()[0])) +"\n")
f.write("DR3 : " + str(numpy.round(len(gok.nonzero()[0])/ areaDR3)) + " "+str(numpy.round(len(rok.nonzero()[0])/ areaDR3)) + " "+str(numpy.round(len(zok.nonzero()[0])/ areaDR3)) +"\n")


plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
gok = (g_mag_dr2>0)&(g_mag_dr2!=numpy.inf)
plt.hist(g_mag_dr2[gok],label="DR2",histtype ='step', bins = binG, weights = numpy.ones_like(g_mag_dr2[gok])/areaDR2)
gok = (g_mag_dr3>0)&(g_mag_dr3!=numpy.inf)
plt.hist(g_mag_dr3[gok],label="DR3",histtype ='step', bins = binG, weights = numpy.ones_like(g_mag_dr3[gok])/areaDR3)
plt.xlabel('g mag')
plt.ylabel('counts per deg2')
plt.xlim((16,30))
plt.legend(loc=2)
plt.yscale('log')
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join(get_outdir('dr23'),"histogram-g-dr2-dr3-f351.png"))
plt.clf()


plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
rok = (r_mag_dr2>0)&(r_mag_dr2!=numpy.inf)
plt.hist(r_mag_dr2[rok],label="DR2",histtype ='step', bins = binG, weights = numpy.ones_like(r_mag_dr2[rok])/areaDR2)
rok = (r_mag_dr3>0)&(r_mag_dr3!=numpy.inf)
plt.hist(r_mag_dr3[rok],label="DR3",histtype ='step', bins = binG, weights = numpy.ones_like(r_mag_dr3[rok])/areaDR3)
plt.xlabel('r mag')
plt.ylabel('counts per deg2')
plt.xlim((16,30))
plt.legend(loc=2)
plt.yscale('log')
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join(get_outdir('dr23'),"histogram-r-dr2-dr3-f351.png"))
plt.clf()



plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
zok = (z_mag_dr2>0)&(z_mag_dr2!=numpy.inf)
plt.hist(z_mag_dr2[zok],label="DR2",histtype ='step', bins = binG, weights = numpy.ones_like(z_mag_dr2[zok])/areaDR2)
zok = (z_mag_dr3>0)&(z_mag_dr3!=numpy.inf)
plt.hist(z_mag_dr3[zok],label="DR3",histtype ='step', bins = binG, weights = numpy.ones_like(z_mag_dr3[zok])/areaDR3)
plt.xlabel('z mag')
plt.ylabel('counts per deg2')
plt.xlim((16,30))
plt.legend(loc=2)
plt.yscale('log')
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join(get_indir('dr23'),"histogram-z-dr2-dr3-f351.png"))
plt.clf()

cFile3 = os.path.join(get_indir('dr23'),"catalog-EDR-242.2-243.7-8-12.1-DR3.fits")
cFile2 = os.path.join(get_indir('dr23'),"catalog-EDR-242.2-243.7-8-12.1-DR2.fits")

hdu = fits.open(cFile3)
dat = hdu[1].data
noJunk3 = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
dr3 = dat[noJunk3]
areaDR3 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )

hdu = fits.open(cFile2)
dat = hdu[1].data
noJunk2 = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
dr2 = dat[noJunk2]
areaDR2 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )

g_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2['decam_flux'].T[1] / dr2['decam_mw_transmission'].T[1])
r_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2['decam_flux'].T[2] / dr2['decam_mw_transmission'].T[2])
z_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2['decam_flux'].T[4] / dr2['decam_mw_transmission'].T[4])
g_e_dr2 = dr2['decam_flux'].T[1] * dr2['decam_flux_ivar'].T[1]**(0.5) 
r_e_dr2 = dr2['decam_flux'].T[2] * dr2['decam_flux_ivar'].T[2]**(0.5) 
z_e_dr2 = dr2['decam_flux'].T[4] * dr2['decam_flux_ivar'].T[4]**(0.5) 

g_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3['decam_flux'].T[1] / dr3['decam_mw_transmission'].T[1])
r_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3['decam_flux'].T[2] / dr3['decam_mw_transmission'].T[2])
z_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3['decam_flux'].T[4] / dr3['decam_mw_transmission'].T[4])
g_e_dr3 = dr3['decam_flux'].T[1] * dr3['decam_flux_ivar'].T[1]**(0.5) 
r_e_dr3 = dr3['decam_flux'].T[2] * dr3['decam_flux_ivar'].T[2]**(0.5) 
z_e_dr3 = dr3['decam_flux'].T[4] * dr3['decam_flux_ivar'].T[4]**(0.5) 

gok = (g_mag_dr2>0)&(g_mag_dr2!=numpy.inf)
rok = (r_mag_dr2>0)&(r_mag_dr2!=numpy.inf)
zok = (z_mag_dr2>0)&(z_mag_dr2!=numpy.inf)
f.write("mags : g ok    r ok   z ok ")
f.write("DR2 : " + str(len(gok.nonzero()[0])) +  " " + str(len(rok.nonzero()[0])) +  " " + str(len(zok.nonzero()[0])) +"\n")
f.write("DR2 : " + str(numpy.round(len(gok.nonzero()[0])/ areaDR2)) +  " " + str(numpy.round(len(rok.nonzero()[0])/ areaDR2)) +  " " + str(numpy.round(len(zok.nonzero()[0])/ areaDR2)) +"\n")
gok = (g_mag_dr3>0)&(g_mag_dr3!=numpy.inf)
rok = (r_mag_dr3>0)&(r_mag_dr3!=numpy.inf)
zok = (z_mag_dr3>0)&(z_mag_dr3!=numpy.inf)
f.write("DR3 : " + str(len(gok.nonzero()[0])) +  " " + str(len(rok.nonzero()[0])) +  " " + str(len(zok.nonzero()[0])) +"\n")
f.write("DR3 : " + str(numpy.round(len(gok.nonzero()[0])/ areaDR3)) + " " + str(numpy.round(len(rok.nonzero()[0])/ areaDR3)) +  " " + str(numpy.round(len(zok.nonzero()[0])/ areaDR3)) +"\n")

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
gok = (g_mag_dr2>0)&(g_mag_dr2!=numpy.inf)
plt.hist(g_mag_dr2[gok],label="DR2",histtype ='step', bins = binG, weights = numpy.ones_like(g_mag_dr2[gok])/areaDR2)
gok = (g_mag_dr3>0)&(g_mag_dr3!=numpy.inf)
plt.hist(g_mag_dr3[gok],label="DR3",histtype ='step', bins = binG, weights = numpy.ones_like(g_mag_dr3[gok])/areaDR3)
plt.xlabel('g mag')
plt.ylabel('counts per deg2')
plt.xlim((16,30))
plt.legend(loc=2)
plt.yscale('log')
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join(get_outdir('dr23'),"histogram-g-dr2-dr3-f242.png"))
plt.clf()


plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
rok = (r_mag_dr2>0)&(r_mag_dr2!=numpy.inf)
plt.hist(r_mag_dr2[rok],label="DR2",histtype ='step', bins = binG, weights = numpy.ones_like(r_mag_dr2[rok])/areaDR2)
rok = (r_mag_dr3>0)&(r_mag_dr3!=numpy.inf)
plt.hist(r_mag_dr3[rok],label="DR3",histtype ='step', bins = binG, weights = numpy.ones_like(r_mag_dr3[rok])/areaDR3)
plt.xlabel('r mag')
plt.ylabel('counts per deg2')
plt.xlim((16,30))
plt.legend(loc=2)
plt.yscale('log')
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join(get_outdir('dr23'),"histogram-r-dr2-dr3-f242.png"))
plt.clf()



plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
zok = (z_mag_dr2>0)&(z_mag_dr2!=numpy.inf)
plt.hist(z_mag_dr2[zok],label="DR2",histtype ='step', bins = binG, weights = numpy.ones_like(z_mag_dr2[zok])/areaDR2)
zok = (z_mag_dr3>0)&(z_mag_dr3!=numpy.inf)
plt.hist(z_mag_dr3[zok],label="DR3",histtype ='step', bins = binG, weights = numpy.ones_like(z_mag_dr3[zok])/areaDR3)
plt.xlabel('z mag')
plt.ylabel('counts per deg2')
plt.xlim((16,30))
plt.legend(loc=2)
plt.yscale('log')
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join(get_indir('dr23'),"histogram-z-dr2-dr3-f242.png"))
plt.clf()

###############################################
# MAthc between dr2 dr3
###############################################


cFile3 = os.path.join(get_indir('dr23'),"catalog-EDR-351.25-353.75-m0.1-0.6-DR3.fits"
cFile2 = os.path.join(get_indir('dr23'),"catalog-EDR-351.25-353.75-m0.1-0.6-DR2.fits"

hdu = fits.open(cFile3)
dat = hdu[1].data
noJunk3 = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
dr3 = dat[noJunk3]
areaDR3 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )

hdu = fits.open(cFile2)
dat = hdu[1].data
noJunk2 = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
dr2 = dat[noJunk2]
areaDR2 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )

dr2T = KDTree(numpy.transpose([dr2['ra'],dr2['dec']]))
dr3T = KDTree(numpy.transpose([dr3['ra'],dr3['dec']]))

# look for the nearest neighbour from dr3 sources in dr2
ds, ids = dr2T.query(dr3T.data)

dsmax = 1/3600.

sel = (ds<dsmax)
len(dr3[sel]) # 116,791 withins 1 arcsec.
len(dr2[ids[sel]])


g_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2[ids[sel]]['decam_flux'].T[1] / dr2[ids[sel]]['decam_mw_transmission'].T[1])
r_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2[ids[sel]]['decam_flux'].T[2] / dr2[ids[sel]]['decam_mw_transmission'].T[2])
z_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2[ids[sel]]['decam_flux'].T[4] / dr2[ids[sel]]['decam_mw_transmission'].T[4])

g_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3[sel]['decam_flux'].T[1] / dr3[sel]['decam_mw_transmission'].T[1])
r_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3[sel]['decam_flux'].T[2] / dr3[sel]['decam_mw_transmission'].T[2])
z_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3[sel]['decam_flux'].T[4] / dr3[sel]['decam_mw_transmission'].T[4])


df_g = dr3[sel]['decam_flux'].T[1] / dr3[sel]['decam_mw_transmission'].T[1] - dr2[ids[sel]]['decam_flux'].T[1] / dr2[ids[sel]]['decam_mw_transmission'].T[1]
sigma_g = (1./dr2[ids[sel]]['decam_flux_ivar'].T[1] + 1./dr3[sel]['decam_flux_ivar'].T[1])**(-0.5)

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(numpy.arange(-10,6,0.1), st.norm.pdf(numpy.arange(-10,6,0.1),loc=0,scale=0.5), 'k--', lw=2, label='N(0,0.5)')
plt.hist(df_g * sigma_g, bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g)/areaDR3, histtype='step', label='all', normed=True)
ok = (g_mag_dr3>20)&(g_mag_dr3<24)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='20<g(DR3)<24', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")&(dr3[sel]['decam_psfsize'].T[1]<1.5)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF, PSFsize<1.5', normed=True)
plt.xlabel('(g(DR3)-g(DR2))/sqrt(var_g(DR2) + var_g(DR3))')
plt.ylabel('Normed counts')
plt.xlim((-2,2))
plt.ylim((0,2))
gp = plt.legend(loc=2, fontsize=10)
gplt.set_frame_on(False)
plt.title('field 351')
plt.grid()
plt.savefig(os.path.join("comparison-depth-normed-g-dr2-dr3-f351.png"))
plt.clf()


df_g = dr3[sel]['decam_flux'].T[2] / dr3[sel]['decam_mw_transmission'].T[2] - dr2[ids[sel]]['decam_flux'].T[2] / dr2[ids[sel]]['decam_mw_transmission'].T[2]
sigma_g = (1./dr2[ids[sel]]['decam_flux_ivar'].T[2] + 1./dr3[sel]['decam_flux_ivar'].T[2])**(-0.5)

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(numpy.arange(-10,6,0.1), st.norm.pdf(numpy.arange(-10,6,0.1),loc=0,scale=0.5), 'k--', lw=2, label='N(0,0.5)')
plt.hist(df_g * sigma_g, bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g)/areaDR3, histtype='step', label='all', normed=True)
ok = (r_mag_dr3>20)&(r_mag_dr3<23)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='20<r(DR3)<23', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")&(dr3[sel]['decam_psfsize'].T[1]<1.5)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF, PSFsize<1.5', normed=True)
plt.xlabel('r(DR3)-r(DR2)/sqrt(var_r(DR2) + var_r(DR3))')
plt.ylabel('counts per deg2')
plt.xlim((-2,2))
plt.ylim((0,2))
gp = plt.legend(loc=2, fontsize=10)
gplt.set_frame_on(False)
plt.title('field 351')
plt.grid()
plt.savefig(os.path.join("comparison-depth-normed-r-dr2-dr3-f351.png"))
plt.clf()



df_g = dr3[sel]['decam_flux'].T[4] / dr3[sel]['decam_mw_transmission'].T[4] - dr2[ids[sel]]['decam_flux'].T[4] / dr2[ids[sel]]['decam_mw_transmission'].T[4]
sigma_g = (1./dr2[ids[sel]]['decam_flux_ivar'].T[4] + 1./dr3[sel]['decam_flux_ivar'].T[4])**(-0.5)

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(numpy.arange(-10,6,0.1), st.norm.pdf(numpy.arange(-10,6,0.1),loc=0,scale=0.5), 'k--', lw=2, label='N(0,0.5)')
plt.hist(df_g * sigma_g, bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g)/areaDR3, histtype='step', label='all', normed=True)
ok = (z_mag_dr3>19)&(z_mag_dr3<22)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='19<z(DR3)<22', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")&(dr3[sel]['decam_psfsize'].T[1]<1.5)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF, PSFsize<1.5', normed=True)
plt.xlabel('z(DR3)-z(DR2)/sqrt(var_z(DR2) + var_z(DR3))')
plt.ylabel('counts per deg2')
plt.xlim((-2,2))
plt.ylim((0,2))
gp = plt.legend(loc=2, fontsize=10)
gplt.set_frame_on(False)
plt.title('field 351')
plt.grid()
plt.savefig(os.path.join("comparison-depth-normed-z-dr2-dr3-f351.png"))
plt.clf()


plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
ok=(dr2[ids[sel]]['type']=="PSF")
plt.plot(g_mag_dr2[ok], g_mag_dr3[ok] - g_mag_dr2[ok], 'b,')
plt.xlabel('g DR2')
plt.ylabel('g DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 351 type PSF')
plt.savefig(os.path.join("comparison-dg-g-psf-dr2-dr3-f351.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(r_mag_dr2[ok], r_mag_dr3[ok]-r_mag_dr2[ok], 'b,')
plt.xlabel('r DR2')
plt.ylabel('r DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 351 type PSF')
plt.savefig(os.path.join("comparison-dr-r-psf-dr2-dr3-f351.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(z_mag_dr2[ok], z_mag_dr3[ok]-z_mag_dr2[ok], 'b,')
plt.xlabel('z DR2')
plt.ylabel('z DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 351 type PSF')
plt.savefig(os.path.join("comparison-dz-z-psf-dr2-dr3-f351.png"))
plt.clf()


plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(g_mag_dr2, g_mag_dr3-g_mag_dr2, 'b,')
plt.xlabel('g DR2')
plt.ylabel('g DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join("comparison-dg-g-zoom-dr2-dr3-f351.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(r_mag_dr2, r_mag_dr3-r_mag_dr2, 'b,')
plt.xlabel('r DR2')
plt.ylabel('r DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join("comparison-dr-r-zoom-dr2-dr3-f351.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(z_mag_dr2, z_mag_dr3-z_mag_dr2, 'b,')
plt.xlabel('z DR2')
plt.ylabel('z DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join("comparison-dz-z-zoom-dr2-dr3-f351.png"))
plt.clf()


plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(g_mag_dr2, g_mag_dr3-g_mag_dr2, 'b,')
plt.xlabel('g DR2')
plt.ylabel('g DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-1,1))
plt.legend(loc=2)
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join("comparison-dg-g-dr2-dr3-f351.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(r_mag_dr2, r_mag_dr3-r_mag_dr2, 'b,')
plt.xlabel('r DR2')
plt.ylabel('r DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-1,1))
plt.legend(loc=2)
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join("comparison-dr-r-dr2-dr3-f351.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(z_mag_dr2, z_mag_dr3-z_mag_dr2, 'b,')
plt.xlabel('z DR2')
plt.ylabel('z DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-1,1))
plt.legend(loc=2)
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join("comparison-dz-z-dr2-dr3-f351.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(z_mag_dr2, z_mag_dr3, 'b,')
plt.plot(z_mag_dr2, z_mag_dr2, 'r,')
plt.xlabel('z DR2')
plt.ylabel('z DR3')
plt.xlim((16,26))
plt.ylim((16,26))
plt.legend(loc=2)
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join("comparison-z-dr2-dr3-f351.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(r_mag_dr2, r_mag_dr3, 'b,')
plt.plot(r_mag_dr2, r_mag_dr2, 'r,')
plt.xlabel('r DR2')
plt.ylabel('r DR3')
plt.xlim((16,26))
plt.ylim((16,26))
plt.legend(loc=2)
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join("comparison-r-dr2-dr3-f351.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(g_mag_dr2, g_mag_dr3, 'b,')
plt.plot(g_mag_dr2, g_mag_dr2, 'r,')
plt.xlabel('g DR2')
plt.ylabel('g DR3')
plt.xlim((16,26))
plt.ylim((16,26))
plt.legend(loc=2)
plt.grid()
plt.title('field 351')
plt.savefig(os.path.join("comparison-g-dr2-dr3-f351.png"))
plt.clf()


###############################################
# MAthc between dr2 dr3
###############################################


cFile3 = "catalog-EDR-242.2-243.7-8-12.1-DR3.fits"
cFile2 = "catalog-EDR-242.2-243.7-8-12.1-DR2.fits"

hdu = fits.open(cFile3)
dat = hdu[1].data
noJunk3 = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
dr3 = dat[noJunk3]
areaDR3 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )

hdu = fits.open(cFile2)
dat = hdu[1].data
noJunk2 = (dat['brick_primary']) & (dat['decam_anymask'].T[1]==0) & (dat['decam_anymask'].T[2]==0) & (dat['decam_anymask'].T[4]==0) 
dr2 = dat[noJunk2]
areaDR2 = ( numpy.max(dat['ra']) - numpy.min(dat['ra']) ) * ( numpy.max(dat['dec']) - numpy.min(dat['dec']) )*numpy.cos( numpy.pi * numpy.mean(dat['dec']) / 180. )

dr2T = KDTree(numpy.transpose([dr2['ra'],dr2['dec']]))
dr3T = KDTree(numpy.transpose([dr3['ra'],dr3['dec']]))

# look for the nearest neighbour from dr3 sources in dr2
ds, ids = dr2T.query(dr3T.data)

dsmax = 1/3600.

sel = (ds<dsmax)
len(dr3[sel]) # 553,948 withins 1 arcsec.
len(dr2[ids[sel]])


g_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2[ids[sel]]['decam_flux'].T[1] / dr2[ids[sel]]['decam_mw_transmission'].T[1])
r_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2[ids[sel]]['decam_flux'].T[2] / dr2[ids[sel]]['decam_mw_transmission'].T[2])
z_mag_dr2 = 22.5 - 2.5 * numpy.log10(dr2[ids[sel]]['decam_flux'].T[4] / dr2[ids[sel]]['decam_mw_transmission'].T[4])

g_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3[sel]['decam_flux'].T[1] / dr3[sel]['decam_mw_transmission'].T[1])
r_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3[sel]['decam_flux'].T[2] / dr3[sel]['decam_mw_transmission'].T[2])
z_mag_dr3 = 22.5 - 2.5 * numpy.log10(dr3[sel]['decam_flux'].T[4] / dr3[sel]['decam_mw_transmission'].T[4])


df_g = dr3[sel]['decam_flux'].T[1] / dr3[sel]['decam_mw_transmission'].T[1] - dr2[ids[sel]]['decam_flux'].T[1] / dr2[ids[sel]]['decam_mw_transmission'].T[1]
sigma_g = (1./dr2[ids[sel]]['decam_flux_ivar'].T[1] + 1./dr3[sel]['decam_flux_ivar'].T[1])**(-0.5)

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(numpy.arange(-10,6,0.1), st.norm.pdf(numpy.arange(-10,6,0.1),loc=0,scale=0.5), 'k--', lw=2, label='N(0,0.5)')
plt.hist(df_g * sigma_g, bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g)/areaDR3, histtype='step', label='all', normed=True)
ok = (g_mag_dr3>20)&(g_mag_dr3<24)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='20<g(DR3)<24', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")&(dr3[sel]['decam_psfsize'].T[1]<1.5)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF, PSFsize<1.5', normed=True)
plt.xlabel('(g(DR3)-g(DR2))/sqrt(var_g(DR2) + var_g(DR3))')
plt.ylabel('Normed counts')
plt.xlim((-2,2))
plt.ylim((0,2))
gp = plt.legend(loc=2, fontsize=10)
gplt.set_frame_on(False)
plt.title('field 242')
plt.grid()
plt.savefig(os.path.join("comparison-depth-normed-g-dr2-dr3-f242.png"))
plt.clf()


df_g = dr3[sel]['decam_flux'].T[2] / dr3[sel]['decam_mw_transmission'].T[2] - dr2[ids[sel]]['decam_flux'].T[2] / dr2[ids[sel]]['decam_mw_transmission'].T[2]
sigma_g = (1./dr2[ids[sel]]['decam_flux_ivar'].T[2] + 1./dr3[sel]['decam_flux_ivar'].T[2])**(-0.5)

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(numpy.arange(-10,6,0.1), st.norm.pdf(numpy.arange(-10,6,0.1),loc=0,scale=0.5), 'k--', lw=2, label='N(0,0.5)')
plt.hist(df_g * sigma_g, bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g)/areaDR3, histtype='step', label='all', normed=True)
ok = (r_mag_dr3>20)&(r_mag_dr3<23)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='20<r(DR3)<23', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")&(dr3[sel]['decam_psfsize'].T[1]<1.5)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF, PSFsize<1.5', normed=True)
plt.xlabel('r(DR3)-r(DR2)/sqrt(var_r(DR2) + var_r(DR3))')
plt.ylabel('counts per deg2')
plt.xlim((-2,2))
plt.ylim((0,2))
gp = plt.legend(loc=2, fontsize=10)
gplt.set_frame_on(False)
plt.title('field 242')
plt.grid()
plt.savefig(os.path.join("comparison-depth-normed-r-dr2-dr3-f242.png"))
plt.clf()



df_g = dr3[sel]['decam_flux'].T[4] / dr3[sel]['decam_mw_transmission'].T[4] - dr2[ids[sel]]['decam_flux'].T[4] / dr2[ids[sel]]['decam_mw_transmission'].T[4]
sigma_g = (1./dr2[ids[sel]]['decam_flux_ivar'].T[4] + 1./dr3[sel]['decam_flux_ivar'].T[4])**(-0.5)

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(numpy.arange(-10,6,0.1), st.norm.pdf(numpy.arange(-10,6,0.1),loc=0,scale=0.5), 'k--', lw=2, label='N(0,0.5)')
plt.hist(df_g * sigma_g, bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g)/areaDR3, histtype='step', label='all', normed=True)
ok = (z_mag_dr3>19)&(z_mag_dr3<22)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='19<z(DR3)<22', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF', normed=True)
ok = (dr2[ids[sel]]['type']=="PSF")&(dr3[sel]['decam_psfsize'].T[1]<1.5)
plt.hist(df_g[ok] * sigma_g[ok], bins=numpy.arange(-10,6,0.1), weights = numpy.ones_like(df_g[ok])/areaDR3, histtype='step', label='type PSF, PSFsize<1.5', normed=True)
plt.xlabel('z(DR3)-z(DR2)/sqrt(var_z(DR2) + var_z(DR3))')
plt.ylabel('counts per deg2')
plt.xlim((-2,2))
plt.ylim((0,2))
gp = plt.legend(loc=2, fontsize=10)
gplt.set_frame_on(False)
plt.title('field 242')
plt.grid()
plt.savefig(os.path.join("comparison-depth-normed-z-dr2-dr3-f242.png"))
plt.clf()




plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
ok=(dr2[ids[sel]]['type']=="PSF")&(dr3[sel]['decam_psfsize'].T[1]<1.5)
plt.plot(g_mag_dr2[ok], g_mag_dr3[ok] - g_mag_dr2[ok], 'b,')
plt.xlabel('g DR2')
plt.ylabel('g DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242 type PSF, g PSFsize<1.5')
plt.savefig(os.path.join("comparison-dg-g-psf-belowMean-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
ok=(dr2[ids[sel]]['type']=="PSF")&(dr3[sel]['decam_psfsize'].T[2]<1.5)
plt.plot(r_mag_dr2[ok], r_mag_dr3[ok]-r_mag_dr2[ok], 'b,')
plt.xlabel('r DR2')
plt.ylabel('r DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242 type PSF, r PSFsize<1.5')
plt.savefig(os.path.join("comparison-dr-r-psf-belowMean-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
ok=(dr2[ids[sel]]['type']=="PSF")&(dr3[sel]['decam_psfsize'].T[4]<1.2)
plt.plot(z_mag_dr2[ok], z_mag_dr3[ok]-z_mag_dr2[ok], 'b,')
plt.xlabel('z DR2')
plt.ylabel('z DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242 type PSF, z PSFsize<1.2')
plt.savefig(os.path.join("comparison-dz-z-psf-belowMean-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
ok=(dr2[ids[sel]]['type']=="PSF")
plt.plot(g_mag_dr2[ok], g_mag_dr3[ok] - g_mag_dr2[ok], 'b,')
plt.xlabel('g DR2')
plt.ylabel('g DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242 type PSF')
plt.savefig(os.path.join("comparison-dg-g-psf-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(r_mag_dr2[ok], r_mag_dr3[ok]-r_mag_dr2[ok], 'b,')
plt.xlabel('r DR2')
plt.ylabel('r DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242 type PSF')
plt.savefig(os.path.join("comparison-dr-r-psf-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(z_mag_dr2[ok], z_mag_dr3[ok]-z_mag_dr2[ok], 'b,')
plt.xlabel('z DR2')
plt.ylabel('z DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242 type PSF')
plt.savefig(os.path.join("comparison-dz-z-psf-dr2-dr3-f242.png"))
plt.clf()


plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(g_mag_dr2, g_mag_dr3-g_mag_dr2, 'b,')
plt.xlabel('g DR2')
plt.ylabel('g DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join("comparison-dg-g-zoom-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(r_mag_dr2, r_mag_dr3-r_mag_dr2, 'b,')
plt.xlabel('r DR2')
plt.ylabel('r DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join("comparison-dr-r-zoom-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(z_mag_dr2, z_mag_dr3-z_mag_dr2, 'b,')
plt.xlabel('z DR2')
plt.ylabel('z DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-0.1,0.1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join("comparison-dz-z-zoom-dr2-dr3-f242.png"))
plt.clf()


plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(g_mag_dr2, g_mag_dr3-g_mag_dr2, 'b,')
plt.xlabel('g DR2')
plt.ylabel('g DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-1,1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join("comparison-dg-g-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(r_mag_dr2, r_mag_dr3-r_mag_dr2, 'b,')
plt.xlabel('r DR2')
plt.ylabel('r DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-1,1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join("comparison-dr-r-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(z_mag_dr2, z_mag_dr3-z_mag_dr2, 'b,')
plt.xlabel('z DR2')
plt.ylabel('z DR3 - DR2')
plt.xlim((16,26))
plt.ylim((-1,1))
plt.legend(loc=2)
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join("comparison-dz-z-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(z_mag_dr2, z_mag_dr3, 'b,')
plt.plot(z_mag_dr2, z_mag_dr2, 'r,')
plt.xlabel('z DR2')
plt.ylabel('z DR3')
plt.xlim((16,26))
plt.ylim((16,26))
plt.legend(loc=2)
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join("comparison-z-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(r_mag_dr2, r_mag_dr3, 'b,')
plt.plot(r_mag_dr2, r_mag_dr2, 'r,')
plt.xlabel('r DR2')
plt.ylabel('r DR3')
plt.xlim((16,26))
plt.ylim((16,26))
plt.legend(loc=2)
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join("comparison-r-dr2-dr3-f242.png"))
plt.clf()

plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
plt.plot(g_mag_dr2, g_mag_dr3, 'b,')
plt.plot(g_mag_dr2, g_mag_dr2, 'r,')
plt.xlabel('g DR2')
plt.ylabel('g DR3')
plt.xlim((16,26))
plt.ylim((16,26))
plt.legend(loc=2)
plt.grid()
plt.title('field 242')
plt.savefig(os.path.join("comparison-g-dr2-dr3-f242.png"))
plt.clf()

f.close()
