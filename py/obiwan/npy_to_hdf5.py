from __future__ import division, print_function
import h5py
from glob import glob
import numpy as np
import csv
import os

def read_dict(fn):
    d = {}
    for key, val in csv.reader(open(fn)):
        d[key] = val
    return d


base='/global/cscratch1/sd/kaylanb/dr3-obiwan/deeplearning'
fobj = h5py.File(os.path.join(base,"test.hdf5"), "a")

# Sources
for obj in ['star','qso','elg','lrg']:
    fns= glob(os.path.join(base,obj,'*/*/rowstart0','*.npy') ) 
    fns= np.sort(fns)
    assert(len(fns) > 0)
    for cnt,fn in enumerate(fns):
        data= np.load(fn)
        band= os.path.basename(fn).split('_')[0]
        id= os.path.basename(fn).split('_')[1]
        ccdname= os.path.basename(fn).split('_')[2].replace('.npy','')
        # Add to hdf5
        node= "/%s/%s/%s" % (id,obj,band)
        if not node in fobj:
            dset = fobj.create_dataset(node, data=data[...,0:2],chunks=True)
            dset.attrs['ccdname']= ccdname
            d= read_dict(fn.replace('.npy','.csv'))
            for key in d.keys():
                dset.attrs[key]= d[key]
        # Backgrounds same for all sources, so just use elg backgrounds
        if obj == 'star':
            node= "/%s/back/%s" % (id,band)
            if not node in fobj:
                dset = fobj.create_dataset(node, data=data[...,2:4],chunks=True)
        if cnt % 20 == 0:
            print('%s: saved 20th npy data set in hdf5' % obj)

print('done')
