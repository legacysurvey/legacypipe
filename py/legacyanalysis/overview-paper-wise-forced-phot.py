
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10,8)
import pylab as plt
from astrometry.util.fits import *
from astrometry.util.plotutils import *
import numpy as np
import fitsio


# In[2]:

cd ~/tractor


# In[3]:

from wise.allwisecat import *


# In[4]:

allwise_catalog_dec_range


# In[5]:

ls /global/homes/d/dstn/cosmo/data/wise/allwise-catalog/


# In[6]:

ls /global/homes/d/dstn/cosmo/data/legacysurvey/dr5/sweep/5.0/sweep-240p005-250p010.fits


# In[7]:

L = fits_table('/global/homes/d/dstn/cosmo/data/legacysurvey/dr5/sweep/5.0/sweep-240p005-250p010.fits')


# In[12]:

print(len(L))
dlo=L.dec.min()
dhi=L.dec.max()
rlo=L.ra.min()
rhi=L.ra.max()
dlo,dhi


# In[11]:

from astrometry.libkd.spherematch import tree_open, tree_search_radec


# In[15]:

W = []
for i,(d1,d2) in enumerate(allwise_catalog_dec_range):
    if d1 < dhi and d2 > dlo:
        print('Overlaps part', i+1)
        #kdfn = '/global/homes/d/dstn/cosmo/data/wise/allwise-catalog/wise-allwise-cat-part%02i-radec.kd' % (i+1)
        #kd = tree_open(kdfn)
        catfn = '/global/homes/d/dstn/cosmo/data/wise/allwise-catalog/wise-allwise-cat-part%02i-radecmpro.fits' % (i+1)
        C = fits_table(catfn)
        print(len(C), 'sources')
        C.cut((C.ra >= rlo) * (C.ra <= rhi) * (C.dec >= dlo) * (C.dec <= dhi))
        print(len(C), 'kept')
        W.append(C)
W = merge_tables(W)


# In[16]:

len(W), len(L)


# In[101]:

from astrometry.libkd.spherematch import match_radec
I,J,d = match_radec(W.ra, W.dec, L.ra, L.dec, 4./3600.)


# In[74]:

len(I)


# In[102]:

from collections import Counter


# In[103]:

CW = Counter(I)
CL = Counter(J)
K, = np.nonzero([(CW[i] == 1) and (CL[j] == 1) for i,j in zip(I,J)])
len(K)


# In[104]:

# Unmatched LS sources
U = np.ones(len(L), bool)
U[J] = False
# Cut to one-to-one unique matches
I = I[K]
J = J[K]


# In[78]:

from astrometry.util.plotutils import *


# In[105]:

L.w1 = -2.5*(np.log10(L.flux_w1)-9.) - 2.699
L.w2 = -2.5*(np.log10(L.flux_w2)-9.) - 3.339
L.w3 = -2.5*(np.log10(L.flux_w3)-9.) - 5.174
L.w4 = -2.5*(np.log10(L.flux_w4)-9.) - 6.620
L.z  = -2.5*(np.log10(L.flux_z)-9.)
L.r  = -2.5*(np.log10(L.flux_r)-9.)

L.is_psf = np.array([t[0]=='P' for t in L.type])

ML = L[J]
MW = W[I]

UL = L[U]
#WISEAB1 =                2.699 / WISE Vega to AB conv for band 1                                                                                                                    
#WISEAB2 =                3.339 / WISE Vega to AB conv for band 2                                                                                                                    
#WISEAB3 =                5.174 / WISE Vega to AB conv for band 3                                                                                                                    
#WISEAB4 =                 6.62 / WISE Vega to AB conv for band 4        


# In[106]:

loghist(MW.w1mpro, ML.w1, 200, range=((5,19),(5,19)), hot=False, imshowargs=dict(cmap=antigray));
ax = plt.axis()
plt.plot([5,21],[5,21], 'k-', alpha=0.2);
plt.axis(ax)
plt.xlabel('WISE W1 mag')
plt.ylabel('Legacy Survey Forced-Photometry W1 mag');
plt.axis([ax[1],ax[0],ax[3],ax[2]]);
plt.savefig('w1-matched.png');


# In[81]:

loghist(ML.z - ML.w1, ML.w1 - ML.w2, 200, range=((-1,5),(-1,5)), hot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('z - W1 (mag)')
plt.ylabel('W1 - W2 (mag)')


# In[83]:

loghist((ML.z - ML.w1)[ML.is_psf], (ML.w1 - ML.w2)[ML.is_psf], 200, range=((-1,5),(-1,5)), hot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('z - W1 (mag)')
plt.ylabel('W1 - W2 (mag)')
plt.title('LegacySurvey PSFs matched to AllWISE catalog')


# In[84]:

plothist((ML.z - ML.w1)[ML.is_psf], (ML.w1 - ML.w2)[ML.is_psf], 200, range=((0.5,3),(-0.5,0.5)), dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('z - W1 (mag)')
plt.ylabel('W1 - W2 (mag)')
plt.title('LegacySurvey PSFs (matched to AllWISE catalog)')


# In[96]:

I = np.logical_not(ML.is_psf)
plothist((ML.z - ML.w1)[I], (ML.w1 - ML.w2)[I], 200, range=((0.5,3),(-0.5,0.5)), dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('z - W1 (mag)')
plt.ylabel('W1 - W2 (mag)')
plt.title('LegacySurvey NON-PSFs (matched to AllWISE catalog)')


# In[98]:

plt.subplot(1,2,1)
I = ML.is_psf
plothist((ML.z - ML.w1)[I], (ML.w1 - ML.w2)[I], 200, range=((0.5,3),(-0.5,0.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('z - W1 (mag)')
plt.ylabel('W1 - W2 (mag)')
plt.title('LegacySurvey PSFs (matched to AllWISE catalog)')

plt.subplot(1,2,2)
I = np.logical_not(ML.is_psf)
plothist((ML.z - ML.w1)[I], (ML.w1 - ML.w2)[I], 200, range=((0.5,3),(-0.5,0.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('z - W1 (mag)')
plt.ylabel('W1 - W2 (mag)')
plt.title('LegacySurvey NON-PSFs (matched to AllWISE catalog)')


# In[95]:

I = ((UL.flux_w1 * np.sqrt(UL.flux_ivar_w1) > 3.) *
     (UL.flux_w2 * np.sqrt(UL.flux_ivar_w2) > 3.) *
     (UL.flux_z  * np.sqrt(UL.flux_ivar_z ) > 3.) *
     (UL.is_psf))
plothist((UL.z - UL.w1)[I], (UL.w1 - UL.w2)[I], 200, range=((0.5,3),(-0.5,0.5)), dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('z - W1 (mag)')
plt.ylabel('W1 - W2 (mag)')
plt.title('LegacySurvey PSFs (UNmatched to AllWISE catalog)')


# In[86]:

plothist((L.z - L.w1)[L.is_psf], (L.w1 - L.w2)[L.is_psf], 200, range=((0.5,3),(-0.5,0.5)), dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('z - W1 (mag)')
plt.ylabel('W1 - W2 (mag)')
plt.title('LegacySurvey PSFs (all)')


# In[70]:

plothist((L.z - L.w1), (L.w1 - L.w2), 200, range=((0.5,3),(-0.5,0.5)), dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('z - W1 (mag)')
plt.ylabel('W1 - W2 (mag)')
plt.title('LegacySurvey (all)')


# In[58]:

I = L.is_psf
loghist((L.z - L.w1)[I], (L.w1 - L.w2)[I], 200, range=((-1,5),(-1,5)), hot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('z - W1 (mag)')
plt.ylabel('W1 - W2 (mag)')


# In[125]:

plt.hist(ML.flux_w1 * np.sqrt(ML.flux_ivar_w1), range=(0,100), bins=100, histtype='step', color='b', log=True);
plt.hist(L.flux_w1 * np.sqrt(L.flux_ivar_w1), range=(0,100), bins=100, histtype='step', color='k', log=True);
plt.hist(UL.flux_w1 * np.sqrt(UL.flux_ivar_w1), range=(0,100), bins=100, histtype='step', color='r', log=True);


# In[ ]:




# In[122]:

plt.hist(ML.w1, range=(10,20), bins=100, histtype='step', color='b', log=True);
plt.hist(L.w1 , range=(10,20), bins=100, histtype='step', color='k', log=True);
plt.hist(UL.w1 , range=(10,20), bins=100, histtype='step', color='r', log=True);
yl,yh = plt.ylim()
plt.ylim(1,yh);


# In[60]:

I = ML.is_psf
plt.hist(ML.flux_w1[I] * np.sqrt(ML.flux_ivar_w1[I]), range=(0,20), bins=100, histtype='step', color='g');
plt.hist(ML.flux_w2[I] * np.sqrt(ML.flux_ivar_w2[I]), range=(0,20), bins=100, histtype='step', color='r');
plt.hist(ML.flux_z[I] *  np.sqrt(ML.flux_ivar_z [I]), range=(0,20), bins=100, histtype='step', color='b');
plt.xlabel('S/N');


# In[130]:

plt.subplot(1,2,1)
I = (ML.is_psf)
plothist((ML.r - ML.z)[I], (ML.z - ML.w1)[I], 200, range=((0,3),(0.5,2.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('r - z (mag)')
plt.ylabel('z - W1 (mag)')
plt.title('LegacySurvey PSFs (matched to AllWISE catalog)')

rz = (ML.r - ML.z)[I]
zw = (ML.z - ML.w1)[I]
print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Matched')

plt.subplot(1,2,2)
I = ((UL.flux_w1 * np.sqrt(UL.flux_ivar_w1) > 5.) *
     (UL.flux_r  * np.sqrt(UL.flux_ivar_r ) > 5.) *
     (UL.flux_z  * np.sqrt(UL.flux_ivar_z ) > 5.) *
     (UL.is_psf))
#I = UL.is_psf
plothist((UL.r - UL.z)[I], (UL.z - UL.w1)[I], 200, range=((0,3),(0.5,2.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('r-z (mag)')
plt.ylabel('z-W1 (mag)')
plt.title('LegacySurvey PSFs (UNmatched to AllWISE catalog)')

rz = (UL.r - UL.z)[I]
zw = (UL.z - UL.w1)[I]
print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Unmatched')

plt.savefig('cc.png')


# In[127]:

plt.subplot(1,2,1)
I = (ML.is_psf)
plothist((ML.r - ML.z)[I], (ML.z - (ML.w1+ML.w2)/2.)[I], 200, range=((0,3),(0.5,2.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('r - z (mag)')
plt.ylabel('z - W (mag)')
plt.title('LegacySurvey PSFs (matched to AllWISE catalog)')

plt.subplot(1,2,2)
I = ((UL.flux_w1 * np.sqrt(UL.flux_ivar_w1) > 3.) *
     (UL.flux_r  * np.sqrt(UL.flux_ivar_r ) > 3.) *
     (UL.flux_z  * np.sqrt(UL.flux_ivar_z ) > 3.) *
     (UL.is_psf))
#I = UL.is_psf
plothist((UL.r - UL.z)[I], (UL.z - (UL.w1+UL.w2)/2.)[I], 200, range=((0,3),(0.5,2.5)), doclf=False, dohot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('r - z (mag)')
plt.ylabel('z - W (mag)')
plt.title('LegacySurvey PSFs (UNmatched to AllWISE catalog)')

#plt.savefig('cc2.png')


# In[128]:

np.sum()


# In[132]:

plt.subplot(1,2,1)
I = (ML.is_psf)
loghist((ML.r - ML.z)[I], (ML.z - ML.w1)[I], 100, range=((0,3),(0.5,2.5)), doclf=False, hot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('r - z (mag)')
plt.ylabel('z - W1 (mag)')
plt.title('LegacySurvey PSFs (matched to AllWISE catalog)')

rz = (ML.r - ML.z)[I]
zw = (ML.z - ML.w1)[I]
print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Matched')

plt.subplot(1,2,2)
I = ((UL.flux_w1 * np.sqrt(UL.flux_ivar_w1) > 3.) *
     (UL.flux_r  * np.sqrt(UL.flux_ivar_r ) > 3.) *
     (UL.flux_z  * np.sqrt(UL.flux_ivar_z ) > 3.) *
     (UL.is_psf))
#I = UL.is_psf
loghist((UL.r - UL.z)[I], (UL.z - UL.w1)[I], 200, range=((0,3),(0.5,2.5)), doclf=False, hot=False, imshowargs=dict(cmap=antigray));
plt.xlabel('r-z (mag)')
plt.ylabel('z-W1 (mag)')
plt.title('LegacySurvey PSFs (UNmatched to AllWISE catalog)')

rz = (UL.r - UL.z)[I]
zw = (UL.z - UL.w1)[I]
print(np.sum((rz>0)*(rz<3)*(zw>0.5)*(zw<2.5)), 'Unmatched')

plt.savefig('cc2.png')


# In[ ]:



