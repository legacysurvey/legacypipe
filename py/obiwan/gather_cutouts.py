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

def dobash(cmd):
    print('UNIX cmd: %s' % cmd)
    if os.system(cmd): raise ValueError

class GatherTraining(object):
    '''1 hdf5 with groups /star,qso,elg,lrg,back each being its own data set'''
    def __init__(self,data_dir=None):
        if data_dir is None:
            self.data_dir= os.getenv('DECALS_SIM_DIR')
        else:
            self.data_dir= data_dir
        self.hdf5_fn= os.path.join(self.data_dir,"training_gathered.hdf5")

    def grzSets(self, obj=None):
        assert(obj in ['elg','lrg','star','qso'])
        get_data=False
        if not os.path.exists(self.hdf5_fn):
            get_data=True
        else:
            fobj= h5py.File(self.hdf5_fn,'r')
            node= '/%s' % obj
            if not node in list( fobj.keys() ):
                get_data=True
        # 
        if not get_data:
            node='/%s' % obj
            print('Already exists: node %s in file %s' % (node,self.hdf5_fn)) 
        else:
            # Wildcard for: elg/138/1381p122/rowstart0/elg_1381p122.hdf5
            fns= glob( os.path.join(self.data_dir,'%s/*/*/rowstart0/%s*.hdf5' % (obj,obj)) )
            assert(len(fns) > 0)
            #n_sets= self.numberGrzSets(fns=fns)
            n_sets= 100 
            # Data Arrays
            print('%s dataset has shape: (%d,64,64,3)' % (obj,n_sets))
            data= np.zeros((n_sets,64,64,3))+np.nan   
            if obj == 'elg': 
                data_back= data.copy() 
            # Fill Data Arrays
            num=0L 
            for fn in fns:
                f= h5py.File(fn,'r')
                for id in list(f.keys()):
                    num_bands= len(f['/%s' % id].keys())
                    if num_bands == 3:
                        # At least one grz set
                        ccdnames={}
                        for band in ['g','r','z']:
                            ccdnames[band]= list( f['/%s/%s' % (id,band)].keys() )
                        num_passes= min([ len(ccdnames['g']),\
                                        len(ccdnames['r']),\
                                        len(ccdnames['z'])])
                        for ith_pass in range(num_passes):
                            for iband,band in zip(range(3),['g','r','z']):
                                    dset= f['/%s/%s/%s' % (id,band,ccdnames[band][ith_pass])]
                                    data[num,:,:,iband]= dset[:,:,0] # Stamp
                                    if obj == 'elg':
                                        data_back[num,:,:,iband]= dset[:,:,1] # Background
                            num += 1
                            if num % 20 == 0:
                                print('grzSets: Read in num=%d' % num)
            # Save
            fobj = h5py.File(self.hdf5_fn, "a") 
            node= '/%s' % obj
            dset = fobj.create_dataset(node, data=data,chunks=True)
            print('Wrote node %s in file %s' % (node,self.hdf5_fn)) 
            if obj == 'elg':
                node= '/%s' % 'back'
                dset = fobj.create_dataset(node, data=data_back,chunks=True)
                print('Wrote node %s in file %s' % (node,self.hdf5_fn)) 
 
    def numberGrzSets(self,fns=None):
        num=0L
        for ith_fn,fn in enumerate(fns):
            f= h5py.File(fn,'r')
            for id in list(f.keys()):
                num_bands= len( list( f['/%s' % id].keys()))
                if num_bands == 3:
                    # At least one grz set
                    num_sets= min([ len( list( f['/%s/g' % id].keys())),\
                                    len( list( f['/%s/r' % id].keys())),\
                                    len( list( f['/%s/z' % id].keys()))])
                    num += num_sets
                    if num % 20 == 0:
                        print('numberGrzSets: num=%d' % num)
        return num

    def readData(self):
        '''return hdf5 file object containing training data'''
        f= h5py.File(self.hdf5_fn,'r')

if __name__ == '__main__':
    gather= GatherTraining()
    for obj in ['elg','qso','star']: #'lrg'
        gather.grzSets(obj= obj)
    f= gather.readData()
    raise ValueError
    print('done')
