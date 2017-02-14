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
    def __init__(self,data_dir=None, rank=0):
        '''
        rank -- optional, use if mpi4py
        '''
        if data_dir is None:
            self.data_dir= os.getenv('DECALS_SIM_DIR')
        else:
            self.data_dir= data_dir
        # Written by some rank
        self.hdf5_fn= os.path.join(self.data_dir,"training_gathered_%d.hdf5" % rank)
        # Final merged product
        self.hdf5_fn_final= os.path.join(self.data_dir,"training_gathered_all.hdf5")

    def get_grzSets(self,fns_dict=None):
        '''
        fns_dict -- dict of fn lists for each objtype, e.g. fns_dict['elg']= list of filename
        '''
        assert(not fns_dict is None)
        for obj in fns_dict.keys():
            assert(obj in ['elg','lrg','star','qso'])
        self.objs= list(fns_dict.keys())
        self.fns_dict= fns_dict
        # 
        for obj in self.objs:
            self.grzSets(obj=obj)

    def grzSets(self, obj=None):
        print('Getting grzSets for obj=%s' % obj)
        # Read/write if exists, create otherwise (default)
        fobj= h5py.File(self.hdf5_fn,'a')
        write_node= '%s' % obj
        # Are we already done?
        if write_node in list( fobj.keys() ):
            print('Already exists: node %s in file %s' % (write_node,self.hdf5_fn)) 
        # Load in the data
        else:
            # Wildcard for: elg/138/1381p122/rowstart0/elg_1381p122.hdf5
            fns= self.fns_dict[obj]
            assert(len(fns) > 0)
            #n_sets= self.numberGrzSets(fns=fns)
            # Data Arrays
            #print('%s dataset has shape: (%d,64,64,3)' % (obj,n_sets))
            data= [] #np.zeros((n_sets,64,64,3))+np.nan   
            if obj == 'star': 
                data_back= [] #data.copy() 
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
                            grz_tmp= np.zeros((64,64,3))+np.nan
                            if obj == 'star':
                                grz_back_tmp = grz_tmp.copy()
                            for iband,band in zip(range(3),['g','r','z']):
                                    dset= f['/%s/%s/%s' % (id,band,ccdnames[band][ith_pass])]
                                    grz_tmp[:,:,iband]= dset[:,:,0] # Stamp
                                    if obj == 'star':
                                        grz_back_tmp[:,:,iband]= dset[:,:,1] # Background
                            data += [grz_tmp]
                            if obj == 'star':
                                data_back += [grz_back_tmp]
                            num += 1
                            if num % 20 == 0:
                                print('grzSets: Read in num=%d' % num)
            # List --> numpy
            data= np.array(data)
            assert(np.all(np.isfinite(data)))
            if obj == 'star':
                data_back= np.array(data_back)
                assert(np.all(np.isfinite(data_back)))
            # 
            print('%s dataset has shape: ' % obj,data.shape)
            # Save
            dset = fobj.create_dataset(write_node, data=data,chunks=True)
            print('Wrote node %s in file %s' % (write_node,self.hdf5_fn)) 
            if obj == 'star':
                node= '%s' % 'back'
                dset = fobj.create_dataset(node, data=data_back,chunks=True)
                print('Wrote node %s in file %s' % (node,self.hdf5_fn)) 

    def merge_rank_data(self):
        '''
        read data in all gather_training_%d.hdf5 files, and store in 
        gather_training_all.hdf5
        '''
        if os.path.exists(self.hdf5_fn_final):
            print('Already exists: %s' % self.hdf5_fn_final)
        else:
            fns= glob( os.path.join(self.data_dir,'training_gathered_*.hdf5') )
            assert(len(fns) > 0)
            all_data={}
            for cnt,fn in enumerate(fns):
                fobj= h5py.File(fn,'r')
                for obj in list(fobj.keys()):
                    assert(obj in ['elg','lrg','star','qso','back'])
                #
                data={}
                for obj in list(fobj.keys()):
                    data[obj]= fobj['/%s' % obj]
                    if cnt == 0:
                        all_data[obj]= data[obj]
                    else:
                        all_data[obj]= np.concatenate((all_data[obj],data[obj]),axis=0)
                print('extracted %d/%d' % (cnt+1,len(fns)))
            # Save 
            f= h5py.File(self.hdf5_fn_final,'w')
            for obj in all_data.keys():
                dset = f.create_dataset(obj, data=all_data[obj],chunks=True)
            print('Wrote %s' % self.hdf5_fn_final)
 
    def readData(self):
        '''return hdf5 file object containing training data'''
        return h5py.File(self.hdf5_fn,'r')

class LookAtTraining(object):
    def __init__(self):

    def bright_examples(self):
        elg['117980298/z/DECam00392483-S20'][...,0].sum()
        elg['117980298/r/DECam00393622-S35'][...,0].sum()
        fn='star.fits';data=star['117980298/z/DECam00392483-S20'];hdu = fits.PrimaryHDU(np.transpose(data));hdulist = fits.HDUList([hdu]);hdu.writeto(fn)
        fn='star_srcback.fits';data=star['117980298/z/DECam00392483-S20'];hdu = fits.PrimaryHDU(data[...,0]+data[...,1]);hdulist = fits.HDUList([hdu]);hdu.writeto(fn)
    def var_examples(self):
    def sum_addednoise_zero(self):
    def starflux_equals_galaxyflux(self):
    def radial_profiles(self):
    


if __name__ == '__main__':
    rank=1
    gather= GatherTraining(rank=rank)
    #fns_dict={}
    #for obj in ['elg','star','qso']:
    #    fns= glob( os.path.join(gather.data_dir,'%s/*/*/rowstart0/%s*.hdf5' % (obj,obj)) )
    #    assert(len(fns) > 0)
    #    fns_dict[obj]= fns[:2]
    #gather.get_grzSets(fns_dict=fns_dict)
    # 
    gather.merge_rank_data()
    raise ValueError
    print('done')
