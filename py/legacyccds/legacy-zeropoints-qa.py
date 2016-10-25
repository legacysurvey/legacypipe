'''
PURPOSE: 
    Compare Arjun's and John's zeropoints files, which are the inputs to Legacypipe/Tractor production runs

CALLING SEQUENCE:
    cd legacypipe/py/legayccds
    python scriptname.py 

REVISION HISTORY:
    25-Oct-2016  K. Burleigh
'''

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import fitsio
import glob
import os

from astrometry.util.fits import fits_table, merge_tables

# Merge tables
# a for Arjun, j for John
fns=glob.glob('/project/projectdirs/desi/users/burleigh/test_data/zeropoint*v2.fits')
j_cat,a_cat= [],[]
for fn in fns:
    j_cat.append( fits_table(fn) )
    a_fn= os.path.join(os.path.dirname(fn), 'arjun_'+os.path.basename(fn))
    a_cat.append( fits_table(a_fn) )
j = merge_tables(j_cat, columns='fillzero')
a = merge_tables(a_cat, columns='fillzero')

# Inspect
print "rows: John=%d, Arjun=%d" % (len(j),len(a))
print "cols: John=%d, Arjun=%d" % (len(j.get_columns()),len(a.get_columns()))

def pset(s):
    print np.sort(list(s))

sj,sa= set(j.get_columns()),set(a.get_columns())
print "Both:\n",pset(sj.intersection(sa))
print "Arjun Only:\n",pset(sa.difference(sj))
print "John Only:\n",pset(sj.difference(sa))

# Plot, keys with SAME NAME
fig,axes= plt.subplots(4,4,figsize=(18,10))
ax=axes.flatten()
hw=0.4
plt.subplots_adjust(hspace=hw,wspace=hw)
cnt=-1
for key in sj.intersection(sa):
    if key in ['date_obs','filter','object','propid','ccdname']:
        continue
    if key in ['ha','ut']:
        continue
    if key in ['zpt']:
        continue #A zpt not -> J zpt
    cnt+=1
    y= (a.get(key) - j.get(key))/a.get(key)
    ax[cnt].scatter(a.get(key),y) 
    xlab=ax[cnt].set_xlabel('A: %s' % key)
    ylab=ax[cnt].set_ylabel('(A-J)/A: %s' % key)
    ax[cnt].set_ylim([-0.01,0.01])
plt.savefig("same_keys.png",\
            bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
plt.close()

# SAME DATA, Different keys
import maps  # mapping between Arjun's and John's keys
fig,axes= plt.subplots(5,4,figsize=(18,10))
ax=axes.flatten()
hw=0.4
plt.subplots_adjust(hspace=hw,wspace=hw)
cnt=-1
# Arjun only
for key in sa.difference(sj):
    #avoid= 
    if key in maps.a_not_in_j():
        continue
    if key in ['filename']:
        continue
    cnt+=1
    if key in ['ccdnmatch','ccdnstar','ccdskycounts']:
        y= j.get( maps.a2j(key) )
        ax[cnt].scatter(a.get(key),y) 
        xlab=ax[cnt].set_xlabel('A: %s' % key)
        ylab=ax[cnt].set_ylabel('J: %s' % maps.a2j(key) )
    else:
        y= (a.get(key) - j.get( maps.a2j(key) ))/a.get(key)
        ax[cnt].scatter(a.get(key),y) 
        xlab=ax[cnt].set_xlabel('A: %s' % key)
        ylab=ax[cnt].set_ylabel('(A-J)/A: %s' % maps.a2j(key) )
    ylim= maps.lims_for_key(key)
    if ylim != 'none':
        ax[cnt].set_ylim([-ylim,ylim])
plt.savefig("same_data_diff_keys.png",\
            bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
plt.close()

# If STRINGS not identicial, print them 
same= ['date_obs','filter','object','propid','ccdname','ha','ut']
mapped= ['filename']
for key in same:
    if a.get(key)[0] != j.get(key)[0]:
        print "Arjuns, Johns (%s)" % key
        for i in range(4): print a.get(key)[i],j.get(key)[i]
for key in mapped:
    if a.get(key)[0] != j.get( maps.a2j(key) )[0]:
        print "Arjuns, Johns (%s)" % key
        for i in range(4): print a.get(key)[i],j.get( maps.a2j(key) )[i]

# Totally DIFFERENT keys, print them
print "Arjun:"
for key in maps.a_not_in_j():
    print "%s: " % key, a.get(key)[0]
    
print "\nJohn:"
for key in maps.j_not_in_a():
    print "%s: " % key, j.get(key)[0]
