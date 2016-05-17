from astrometry.util.fits import fits_table,merge_tables
import numpy as np
from argparse import ArgumentParser
import os

def row_of_ccd(table,ccd,fn):
    '''given a fits_table "table", return row where ccd number is "ccd" filename is "fn"'''
    i_fn= table.get('image_filename') == fn
    i_ccd= table.get('ccdnum').astype(str) == str(ccd) #compare strings
    index= np.all((i_fn,i_ccd),axis=0)
    assert(np.any(index)) #at least one index must be True
    return index 

parser = ArgumentParser(description="test")
parser.add_argument("-ians",action="store",help='Ians bok ccd table',required=True)
parser.add_argument("-simple",action="store",help='table from simple-bok-ccd.py',required=True)
parser.add_argument("-pixscale",type=float,default=0.445,action="store",help='bok',required=False)
args = parser.parse_args()

ians=fits_table(args.ians)
simple=fits_table(args.simple)
#initialize legacypipe table as simple
comb=fits_table(args.simple) 
#put 0s in fields we will grab from ians table
cols= ['ccdzpt','ccdzpta','ccdzptb','ccdphrms','ccdraoff','ccddecoff','ccdnstar','ccdnmatch','ccdnmatcha','ccdnmatchb','ccdmdncol']
cols+= ['seeing','arawgain','avsky','mjd_obs','expnum']
for col in cols: comb.set(col, np.zeros(simple.get('ra').shape).astype(ians.get(col).dtype) ) 
#loop over filenames,ccds 1-4 
for fn in simple.get('image_filename'):
    fn= os.path.basename(fn)
    for ccd in range(1,5):
        #index where ccd,fn occurs    
        i_ians= row_of_ccd(ians,ccd,fn)
        i_simp= row_of_ccd(simple,ccd,'90prime/'+fn) #fn prefixed with 90prime
        #copy over ian's table info
        for col in cols: 
            comb.get(col)[i_simp]= ians.get(col)[i_ians]
            if 'zpt' in col: comb.get(col)[i_simp]+= 2.5*np.log10(simple.get('exptime')[i_simp]) #correct for exptime
            #ian has seeing, also store fwhm 
            comb.get('fwhm')[i_simp]= ians.get('seeing')[i_ians] / args.pixscale
#write new table
comb.writeto('bok-ccds-for-legacypipe.fits')
print 'wrote legacypipe ccd table'
