from astrometry.util.fits import fits_table,merge_tables
import numpy as np

i=fits.open('bass-ccds-idmnov2015.fits')
k=fits.open('my_bok-ccds.fits')
#matching indices for g,r ccds
k_test,i_test=[],[]
for band in ['g','r']:
    #images 1-6 
    for cnt in range(1,7):
        k_test+= list( np.where(k.get('image_filename') == 'bok/deep2%s_ra352_%d.fits' % (band,cnt))[0] )
        i_test+= list( np.where(i.get('image_filename') == 'deep2%s_ra352_%d.fits' % (band,cnt))[0] )
#print line by line, they should match
for ik,ii in zip(k_test,i_test): print k.get('image_filename')[ik],k.get('ccdname')[ik],i.get('image_filename')[ii],i.get('ccdname')[ii]
#make copy of my ccds table
final=fits_table()
for k_c in k.get_columns(): final.set(k_c,k.get(k_c)[k_test])
#insert new vals from ians table
for c in ['arawgain','avsky','mjd_obs','expnum','ccdzpt']: final.set(c,i.get(c)[i_test])
#write new table
final.writeto('kaylan+ian-bok-ccds.fits')
print 'done'
