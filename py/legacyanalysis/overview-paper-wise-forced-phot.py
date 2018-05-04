import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import pylab as plt
from astrometry.util.fits import *
from astrometry.util.plotutils import *
import numpy as np
import fitsio
from glob import glob
from wise.allwisecat import *

plt.figure(figsize=(5,4))

plt.subplots_adjust(right=0.95, top=0.98)

np.errstate(all='ignore')

# Read DR5 LegacySurvey catalogs
#L = fits_table('/global/homes/d/dstn/cosmo/data/legacysurvey/dr5/sweep/5.0/sweep-240p005-250p010.fits')
#fns = ['/global/homes/d/dstn/cosmo/data/legacysurvey/dr5/sweep/5.0/sweep-240p005-250p010.fits']
fns = glob('/global/project/projectdirs/cosmo/data/legacysurvey/dr5/sweep/5.0/sweep-[12]*p005-*p010.fits')
L = []
for fn in fns:
    print('Reading', fn)
    L.append(fits_table(fn, columns=['ra','dec','type',
                                     'flux_g','flux_r','flux_z',
                                     'flux_w1','flux_w2','flux_w3', 'flux_w4',
                                     'flux_ivar_g','flux_ivar_r', 'flux_ivar_z',
                                     'flux_ivar_w1','flux_ivar_w2',
                                     'flux_ivar_w3', 'flux_ivar_w4',
                                     'mw_transmission_g','mw_transmission_r',
                                     'mw_transmission_z',
                                     'mw_transmission_w1','mw_transmission_w2',
                                     'mw_transmission_w3', 'mw_transmission_w4',]))
L = merge_tables(L)

print(len(L), 'LegacySurvey sources')
L.cut((L.ra > 120) * (L.ra < 250))
print('Cut to', len(L), 'in RA 120-250')

L.writeto('/global/cscratch1/sd/dstn/ls.fits')

dlo=L.dec.min()
dhi=L.dec.max()
rlo=L.ra.min()
rhi=L.ra.max()
print('RA', rlo,rhi, 'Dec', dlo,dhi)

# Read AllWISE catalog
W = []
for i,(d1,d2) in enumerate(allwise_catalog_dec_range):
    if d1 < dhi and d2 > dlo:
        print('Overlaps part', i+1)
        catfn = '/global/homes/d/dstn/cosmo/data/wise/allwise-catalog/wise-allwise-cat-part%02i-radecmpro.fits' % (i+1)
        C = fits_table(catfn)
        print(len(C), 'sources')
        C.cut((C.ra >= rlo) * (C.ra <= rhi) * (C.dec >= dlo) * (C.dec <= dhi))
        print(len(C), 'kept')
        W.append(C)
W = merge_tables(W)
print(len(W), 'AllWISE catalog sources')

W.writeto('/global/cscratch1/sd/dstn/wise.fits')

from astrometry.libkd.spherematch import match_radec
print('Matching...')
I,J,d = match_radec(W.ra, W.dec, L.ra, L.dec, 4./3600.)
print(len(I), 'matches')

from collections import Counter

CW = Counter(I)
CL = Counter(J)
K, = np.nonzero([(CW[i] == 1) and (CL[j] == 1) for i,j in zip(I,J)])
print(len(K), 'unique matches')

# Unmatched LS sources
U = np.ones(len(L), bool)
U[J] = False
# Cut to one-to-one unique matches
I = I[K]
J = J[K]

# Compute mags, un-applying the Vega-to-AB conversion factors
L.w1 = -2.5*(np.log10(L.flux_w1)-9.) - 2.699
L.w2 = -2.5*(np.log10(L.flux_w2)-9.) - 3.339
L.w3 = -2.5*(np.log10(L.flux_w3)-9.) - 5.174
L.w4 = -2.5*(np.log10(L.flux_w4)-9.) - 6.620
L.z  = -2.5*(np.log10(L.flux_z)-9.)
L.r  = -2.5*(np.log10(L.flux_r)-9.)

L.e_r = 2.5 * np.log10(L.mw_transmission_r)
L.e_z = 2.5 * np.log10(L.mw_transmission_z)
L.e_w1 = 2.5 * np.log10(L.mw_transmission_w1)
L.e_w2 = 2.5 * np.log10(L.mw_transmission_w2)
L.e_w3 = 2.5 * np.log10(L.mw_transmission_w3)
L.e_w4 = 2.5 * np.log10(L.mw_transmission_w4)

L.is_psf = np.array([t[0]=='P' for t in L.type])

# Matched
ML = L[J]
MW = W[I]
# Unmatched
UL = L[U]

#WISEAB1 =                2.699 / WISE Vega to AB conv for band 1
#WISEAB2 =                3.339 / WISE Vega to AB conv for band 2
#WISEAB3 =                5.174 / WISE Vega to AB conv for band 3
#WISEAB4 =                 6.62 / WISE Vega to AB conv for band 4

loghist(MW.w1mpro, ML.w1, 200, range=((5,19),(5,19)), hot=False, imshowargs=dict(cmap=antigray))
ax = plt.axis()
plt.plot([5,21],[5,21], 'k-', alpha=0.2)
plt.axis(ax)
plt.xlabel('AllWISE W1 mag')
plt.ylabel('Legacy Survey Forced-Photometry W1 mag')
plt.axis([ax[1],ax[0],ax[3],ax[2]])
plt.savefig('w1-matched.pdf')


plt.clf()
lo,hi = 10,23
ha=dict(range=(lo,hi), bins=150, histtype='step', color='b', log=True)
n,b,p1 = plt.hist(W.w1mpro, **ha)
n,b,p2 = plt.hist(L.w1, lw=3, alpha=0.4, **ha)
plt.legend((p1[0],p2[0]), ('AllWISE Catalog', 'LegacySurvey Forced'),
           loc='lower left')
plt.xlim(lo,hi)
yl,yh = plt.ylim()
print('Plot limits:', yl,yh)
plt.ylim(10,yh)
#plt.ylim(10,1e5)
plt.xlabel('W1 mag')
plt.ylabel('Number of sources')
plt.savefig('w1-count.pdf')




plt.clf()
I = (ML.is_psf)
ha = dict(nbins=100, range=((0,2.5),(0.5,3)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray),
          docolorbar=False)
H,xe,ye = plothist((ML.r - ML.z)[I], (ML.z - ML.w1)[I], **ha)
plt.xlabel('r - z (mag)')
plt.ylabel('z - W1 (mag)')
#plt.title('Catalog-matched PSFs')
plt.savefig('cc-matched.pdf')
print(np.sum(H), 'matched')
# rz = (ML.r - ML.z)[I]
# zw = (ML.z - ML.w1)[I]
# print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Matched')

plt.clf()
I = ((UL.flux_w1 * np.sqrt(UL.flux_ivar_w1) > 3.) *
     (UL.flux_r  * np.sqrt(UL.flux_ivar_r ) > 5.) *
     (UL.flux_z  * np.sqrt(UL.flux_ivar_z ) > 5.) *
     (UL.is_psf))
H,xe,ye = plothist((UL.r - UL.z)[I], (UL.z - UL.w1)[I], **ha)
plt.xlabel('r - z (mag)')
plt.ylabel('z - W1 (mag)')
plt.savefig('cc-unmatched.pdf')
#plt.title('LegacySurvey PSF without AllWISE counterparts')
#plt.title('Additional faint PSF sources')
print(np.sum(H), 'matched')


# rz = (UL.r - UL.z)[I]
# zw = (UL.z - UL.w1)[I]
# print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Unmatched')
# plt.savefig('cc.png')



# loghist(ML.z - ML.w1, ML.w1 - ML.w2, 200, range=((-1,5),(-1,5)), hot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('z - W1 (mag)')
# plt.ylabel('W1 - W2 (mag)')
# 
# loghist((ML.z - ML.w1)[ML.is_psf], (ML.w1 - ML.w2)[ML.is_psf], 200, range=((-1,5),(-1,5)), hot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('z - W1 (mag)')
# plt.ylabel('W1 - W2 (mag)')
# plt.title('LegacySurvey PSFs matched to AllWISE catalog')
# 
# plothist((ML.z - ML.w1)[ML.is_psf], (ML.w1 - ML.w2)[ML.is_psf], 200, range=((0.5,3),(-0.5,0.5)), dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('z - W1 (mag)')
# plt.ylabel('W1 - W2 (mag)')
# plt.title('LegacySurvey PSFs (matched to AllWISE catalog)')
# 
# I = np.logical_not(ML.is_psf)
# plothist((ML.z - ML.w1)[I], (ML.w1 - ML.w2)[I], 200, range=((0.5,3),(-0.5,0.5)), dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('z - W1 (mag)')
# plt.ylabel('W1 - W2 (mag)')
# plt.title('LegacySurvey NON-PSFs (matched to AllWISE catalog)')
# 
# plt.subplot(1,2,1)
# I = ML.is_psf
# plothist((ML.z - ML.w1)[I], (ML.w1 - ML.w2)[I], 200, range=((0.5,3),(-0.5,0.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('z - W1 (mag)')
# plt.ylabel('W1 - W2 (mag)')
# plt.title('LegacySurvey PSFs (matched to AllWISE catalog)')
# 
# plt.subplot(1,2,2)
# I = np.logical_not(ML.is_psf)
# plothist((ML.z - ML.w1)[I], (ML.w1 - ML.w2)[I], 200, range=((0.5,3),(-0.5,0.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('z - W1 (mag)')
# plt.ylabel('W1 - W2 (mag)')
# plt.title('LegacySurvey NON-PSFs (matched to AllWISE catalog)')

# I = ((UL.flux_w1 * np.sqrt(UL.flux_ivar_w1) > 3.) *
#      (UL.flux_w2 * np.sqrt(UL.flux_ivar_w2) > 3.) *
#      (UL.flux_z  * np.sqrt(UL.flux_ivar_z ) > 3.) *
#      (UL.is_psf))
# plothist((UL.z - UL.w1)[I], (UL.w1 - UL.w2)[I], 200, range=((0.5,3),(-0.5,0.5)), dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('z - W1 (mag)')
# plt.ylabel('W1 - W2 (mag)')
# plt.title('LegacySurvey PSFs (UNmatched to AllWISE catalog)')
# 
# 
# # In[86]:
# 
# plothist((L.z - L.w1)[L.is_psf], (L.w1 - L.w2)[L.is_psf], 200, range=((0.5,3),(-0.5,0.5)), dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('z - W1 (mag)')
# plt.ylabel('W1 - W2 (mag)')
# plt.title('LegacySurvey PSFs (all)')
# 
# 
# # In[70]:
# 
# plothist((L.z - L.w1), (L.w1 - L.w2), 200, range=((0.5,3),(-0.5,0.5)), dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('z - W1 (mag)')
# plt.ylabel('W1 - W2 (mag)')
# plt.title('LegacySurvey (all)')
# 
# 
# # In[58]:
# 
# I = L.is_psf
# loghist((L.z - L.w1)[I], (L.w1 - L.w2)[I], 200, range=((-1,5),(-1,5)), hot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('z - W1 (mag)')
# plt.ylabel('W1 - W2 (mag)')
# 
# 
# # In[125]:
# 
# plt.hist(ML.flux_w1 * np.sqrt(ML.flux_ivar_w1), range=(0,100), bins=100, histtype='step', color='b', log=True);
# plt.hist(L.flux_w1 * np.sqrt(L.flux_ivar_w1), range=(0,100), bins=100, histtype='step', color='k', log=True);
# plt.hist(UL.flux_w1 * np.sqrt(UL.flux_ivar_w1), range=(0,100), bins=100, histtype='step', color='r', log=True);
# 
# 
# # In[ ]:
# 
# 
# 
# 
# # In[122]:
# 
# plt.hist(ML.w1, range=(10,20), bins=100, histtype='step', color='b', log=True);
# plt.hist(L.w1 , range=(10,20), bins=100, histtype='step', color='k', log=True);
# plt.hist(UL.w1 , range=(10,20), bins=100, histtype='step', color='r', log=True);
# yl,yh = plt.ylim()
# plt.ylim(1,yh);
# 
# 
# # In[60]:
# 
# I = ML.is_psf
# plt.hist(ML.flux_w1[I] * np.sqrt(ML.flux_ivar_w1[I]), range=(0,20), bins=100, histtype='step', color='g');
# plt.hist(ML.flux_w2[I] * np.sqrt(ML.flux_ivar_w2[I]), range=(0,20), bins=100, histtype='step', color='r');
# plt.hist(ML.flux_z[I] *  np.sqrt(ML.flux_ivar_z [I]), range=(0,20), bins=100, histtype='step', color='b');
# plt.xlabel('S/N');
# 
# 
# # In[130]:
# 
# plt.subplot(1,2,1)
# I = (ML.is_psf)
# plothist((ML.r - ML.z)[I], (ML.z - ML.w1)[I], 200, range=((0,3),(0.5,2.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('r - z (mag)')
# plt.ylabel('z - W1 (mag)')
# plt.title('LegacySurvey PSFs (matched to AllWISE catalog)')
# 
# rz = (ML.r - ML.z)[I]
# zw = (ML.z - ML.w1)[I]
# print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Matched')
# 
# plt.subplot(1,2,2)
# I = ((UL.flux_w1 * np.sqrt(UL.flux_ivar_w1) > 5.) *
#      (UL.flux_r  * np.sqrt(UL.flux_ivar_r ) > 5.) *
#      (UL.flux_z  * np.sqrt(UL.flux_ivar_z ) > 5.) *
#      (UL.is_psf))
# #I = UL.is_psf
# plothist((UL.r - UL.z)[I], (UL.z - UL.w1)[I], 200, range=((0,3),(0.5,2.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('r-z (mag)')
# plt.ylabel('z-W1 (mag)')
# plt.title('LegacySurvey PSFs (UNmatched to AllWISE catalog)')
# 
# rz = (UL.r - UL.z)[I]
# zw = (UL.z - UL.w1)[I]
# print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Unmatched')
# 
# plt.savefig('cc.png')
# 
# 
# # In[127]:
# 
# plt.subplot(1,2,1)
# I = (ML.is_psf)
# plothist((ML.r - ML.z)[I], (ML.z - (ML.w1+ML.w2)/2.)[I], 200, range=((0,3),(0.5,2.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('r - z (mag)')
# plt.ylabel('z - W (mag)')
# plt.title('LegacySurvey PSFs (matched to AllWISE catalog)')
# 
# plt.subplot(1,2,2)
# I = ((UL.flux_w1 * np.sqrt(UL.flux_ivar_w1) > 3.) *
#      (UL.flux_r  * np.sqrt(UL.flux_ivar_r ) > 3.) *
#      (UL.flux_z  * np.sqrt(UL.flux_ivar_z ) > 3.) *
#      (UL.is_psf))
# #I = UL.is_psf
# plothist((UL.r - UL.z)[I], (UL.z - (UL.w1+UL.w2)/2.)[I], 200, range=((0,3),(0.5,2.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
# plt.xlabel('r - z (mag)')
# plt.ylabel('z - W (mag)')
# plt.title('LegacySurvey PSFs (UNmatched to AllWISE catalog)')

#plt.subplot(1,2,1)
if False:
    plt.clf()
    ha = dict(nbins=100, range=((-0.5,3),(0,3)), doclf=False, hot=False, imshowargs=dict(cmap=antigray))
    I = (ML.is_psf)
    loghist((ML.r - ML.z)[I], (ML.z - ML.w1)[I], **ha)
    plt.xlabel('r - z (mag)')
    plt.ylabel('z - W1 (mag)')
    #plt.title('LegacySurvey PSFs matched to AllWISE catalog')
    plt.savefig('cc-matched.pdf')
    rz = (ML.r - ML.z)[I]
    zw = (ML.z - ML.w1)[I]
    print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Matched')
    
    plt.clf()
    ha.update(imshowargs=dict(cmap=antigray, vmax=np.log10(3000)))
    I = ((UL.flux_w1 * np.sqrt(UL.flux_ivar_w1) > 3.) *
         (UL.flux_r  * np.sqrt(UL.flux_ivar_r ) > 3.) *
         (UL.flux_z  * np.sqrt(UL.flux_ivar_z ) > 3.) *
         (UL.is_psf))
    loghist((UL.r - UL.z)[I], (UL.z - UL.w1)[I], **ha)
    plt.xlabel('r - z (mag)')
    plt.ylabel('z - W1 (mag)')
    #plt.title('LegacySurvey PSFs unmatched to AllWISE catalog')
    plt.savefig('cc-unmatched.pdf')
    rz = (UL.r - UL.z)[I]
    zw = (UL.z - UL.w1)[I]
    print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Unmatched')
    
    
    plt.clf()
    ha = dict(nbins=100, range=((-0.5,3),(0,3)), doclf=False, hot=False, imshowargs=dict(cmap=antigray))
    I = (ML.is_psf)
    rz = ((ML.r-ML.e_r) - (ML.z-ML.e_z))[I]
    zw = ((ML.z-ML.e_z) - (ML.w1-ML.e_w1))[I]
    loghist(rz, zw, **ha)
    plt.xlabel('r - z (mag)')
    plt.ylabel('z - W1 (mag)')
    #plt.title('LegacySurvey PSFs matched to AllWISE catalog')
    plt.savefig('cc-matched2.pdf')
    print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Matched')
    
    plt.clf()
    ha.update(imshowargs=dict(cmap=antigray, vmax=np.log10(3000)))
    I = ((UL.flux_w1 * np.sqrt(UL.flux_ivar_w1) > 3.) *
         (UL.flux_r  * np.sqrt(UL.flux_ivar_r ) > 3.) *
         (UL.flux_z  * np.sqrt(UL.flux_ivar_z ) > 3.) *
         (UL.is_psf))
    rz = ((UL.r-UL.e_r) - (UL.z-UL.e_z))[I]
    zw = ((UL.z-UL.e_z) - (UL.w1-UL.e_w1))[I]
    loghist(rz, zw, **ha)
    plt.xlabel('r - z (mag)')
    plt.ylabel('z - W1 (mag)')
    #plt.title('LegacySurvey PSFs unmatched to AllWISE catalog')
    plt.savefig('cc-unmatched2.pdf')
    print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Unmatched')


plt.clf()
ha = dict(nbins=200, range=((-5,10),(13,25)), doclf=False, hot=False, imshowargs=dict(cmap=antigray, vmax=4.))
I = (ML.is_psf)
loghist((ML.r - ML.w1)[I], ML.r[I], **ha)
plt.xlabel('r - W1 (mag)')
plt.ylabel('r (mag)')
#plt.title('LegacySurvey PSFs (matched to AllWISE catalog)')
plt.savefig('cm-matched.pdf')

plt.clf()
I = (#(L.flux_w1 * np.sqrt(L.flux_ivar_w1) > 3.) *
     #(L.flux_r  * np.sqrt(L.flux_ivar_r ) > 3.) *
     #(L.flux_z  * np.sqrt(L.flux_ivar_z ) > 3.) *
     (L.is_psf))
loghist((L.r - L.w1)[I], L.r[I], **ha)
plt.xlabel('r - W1 (mag)')
plt.ylabel('r (mag)')
plt.savefig('cm-all.pdf')
