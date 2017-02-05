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
    def __init__(self,obj=None):
        assert(obj in ['elg','lrg','star','qso'])
        self.obj= obj
        self.base= '/global/cscratch1/sd/kaylanb/dr3-obiwan/deeplearning'

    def grzSets(self):
        all_files= '%s_training.txt' % self.obj
        uniq_files= '%s_traning_uniqe.txt' % self.obj
        hasgrz_files= '%s_traning_hasgrz.txt' % self.obj
        # List all .npy artificial source array files
        if not os.path.exists(all_files):
            dobash("find %s/%s/122/*/rowstart0/*.npy > %s" % \
                    (self.base,self.obj,all_files))
            print('Wrote %s' % all_files)
        # List of unique "brickname ID" pairs
        if not os.path.exists(uniq_files):
            dobash("cat %s|awk -F '/' '{print $10,$NF}'|sed s/\ [grz]_/_/g|awk -F '_' '{print $1,$2}'|uniq > %s" % (all_files,uniq_files) )
            print('Wrote %s' % uniq_files)
        # List fns for all "brickname ID" pairs with grz=1 
        bricks,ids=np.loadtxt(uniq_files, dtype=str,unpack=True)
        if os.path.exists(hasgrz_files):
            print('Already exists: %s' % hasgrz_files)
        else:
            print('Gathering grz sets for %s' % self.obj)
            with open(hasgrz_files,'w') as foo:
                for cnt,brick,id in zip(range(len(bricks)),bricks,ids):
                    if cnt % 100 == 0:
                        print('Uniq brickname,id pair: %d/%d' % (cnt,len(bricks)))
                    fns={}
                    for band in ['g','r','z']:
                        fn=os.path.join(self.base,self.obj,brick[:3],brick,\
                                        'rowstart0','%s_%s*.npy' % (band,id))
                        fns[band]= np.array( glob(fn) )
                    sz= min([fns['g'].size,\
                             fns['r'].size,\
                             fns['z'].size])
                    if sz > 0:
                        for i in range(sz):
                            foo.write('%s %s %s\n' % (fns['g'][i],fns['r'][i],fns['z'][i]))
                        if sz > 1: 
                            print('More than 1 grz set!, %d sets' % sz)
            print('Wrote: %s' % hasgrz_files)
        self.hasgrz_files= hasgrz_files
       
    def loadHDF5(self):
        self.hasgrz_files_10k= '%s_traning_hasgrz_10k.txt' % self.obj 
        dobash('head -n 2000 %s > %s' % (self.hasgrz_files,self.hasgrz_files_10k))

        self.hdf5_fn= os.path.join(self.base,"training_10k.hdf5")
        fobj = h5py.File(self.hdf5_fn, "a")
        # Sources
        #for obj in ['star','qso','elg','lrg']:
        #fns= glob(os.path.join(base,obj,'*/*/rowstart0','*.npy') ) 
        #fns= np.sort(fns)
        #assert(len(fns) > 0)

        g_fns,r_fns,z_fns= np.loadtxt(self.hasgrz_files_10k,dtype=str,unpack=True)
        for cnt,g_fn,r_fn,z_fn in zip(range(len(g_fns)), g_fns,r_fns,z_fns):
            id= os.path.basename(g_fn).split('_')[1]
            #ccdname= os.path.basename(fn).split('_')[2].replace('.npy','')
            node= "/%s/%s" % (id,self.obj)
            # Only load if node doesn't exist
            if not node in fobj:
                # 200x200x3 array of grz src
                data= self.read_grz(g_fn,r_fn,z_fn)
                # Add to hdf5
                dset = fobj.create_dataset(node, data=data['src'],chunks=True)
                #dset.attrs['ccdname']= ccdname
                #d= read_dict(fn.replace('.npy','.csv'))
                #for key in d.keys():
                #    dset.attrs[key]= d[key]
            # Backgrounds same for all sources, so just use elg backgrounds
            if self.obj == 'elg':
                node= "/%s/back" % (id,)
                if not node in fobj:
                    dset = fobj.create_dataset(node, data=data['back'],chunks=True)
            if cnt % 100 == 0:
                print('%s data set loaded into hdf5: %d/%d' % (self.obj,cnt,len(g_fns)))

    def read_grz(self,g_fn,r_fn,z_fn):
        ''' data[...,0:2] -- src img,invvar
            data[...,2:4] -- back img,invvar
            data[...,4:6] -- src+back img,invvar
        '''
        # Confirm bands are what say they are
        for band,fn in zip(['g','r','z'],[g_fn,r_fn,z_fn]):
            assert(os.path.basename(fn).split('_')[0] == band)
        # Put g, then r, then z data in 200x200x3 array
        grz=dict(src= np.zeros((200,200,3)).astype(np.float32)+np.nan,\
                 back= np.zeros((200,200,3)).astype(np.float32)+np.nan)
        for i,fn in enumerate([g_fn,r_fn,z_fn]):
            data= np.load(fn)
            # src grz
            grz['src'][:,:,i]= data[...,0] 
            # back grz
            grz['back'][:,:,i]= data[...,2] 
        return grz   

if __name__ == '__main__':
    gather= {}
    for obj in ['elg','lrg','qso','star']:
        gather[obj]= GatherTraining(obj=obj)
        gather[obj].grzSets()
        gather[obj].loadHDF5()

    print('done')
