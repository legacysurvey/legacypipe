#!/usr/bin/env python

"""
Running eBOSS NGC, SGC
kaylan Edison scratch
LEGACY_SURVEY_DIR=/scratch2/scratchdirs/kaylanb/dr3-obiwan/legacypipe-dir
DECALS_SIM_DIR=/scratch2/scratchdirs/kaylanb/obiwan-eboss-ngc


python legacypipe/queue-calibs.py --ignore_cuts --touching --save_to_fits --name eboss-sgc --region eboss-sgc --bricks /global/project/projectdirs/cosmo/data/legacysurvey/dr3//survey-bricks-dr3.fits.gz --ccds /scratch1/scratchdirs/desiproc/DRs/dr4-bootes/legacypipe-dir/survey-ccds-decals-extra-nondecals.fits.gz

ngc=fits_table('bricks-eboss-ngc-cut.fits')
with open('bricks-eboss-ngc.txt','w') as foo:
    for brick in ngc.brickname:
        foo.write('%s\n' % brick)
foo.close()

ccds=fits_table('ccds-eboss-ngc-cut.fits')
ccds.writeto('survey-ccds-dr3-eboss-ngc.fits.gz')
ccds.image_filename= np.char.replace(ccds.image_filename,'decam/','')
fns=np.array( list( set(ccds.image_filename) ) )
with open('fns-eboss-ngc.txt','w') as foo:
   for fn in fns:
       foo.write('%s\n' % fn)
foo.close()

for fn in `cat fns-eboss-ngc.txt`;do find /project/projectdirs/cosmo/staging/decam -type f -name $(basename $fn) >> full-fns-eboss-ngc.txt ;done

cp full-fns-eboss-ngc.txt full-fns-eboss-ngc_wild.txt
sed -i s/_ooi_/_*_/g full-fns-eboss-ngc_wild.txt
sed -i s/_oki_/_*_/g full-fns-eboss-ngc_wild.txt

for fn in `cat full-fns-eboss-ngc_wild.txt`; do rsync -Riv -rlpgoD --size-only $fn /scratch1/scratchdirs/desiproc/DRs/cp-images/decam/;done
# divide filelist into N files each having the next 1000 lines
i=0,j=1;for cnt in `seq 0 8`;do let i=1+$cnt*1000; let j=$i+1000;sed -n ${i},${j}p fns-eboss-sgc_wildcard.txt > fns-eboss-sgc_wildcard_${cnt}.txt;done

LEGACY_SURVEY_DIR
mkdir -p images/decam
cd images/decam
for fullnm in `find /scratch1/scratchdirs/desiproc/DRs/cp-images/decam/project/projectdirs/cosmo/staging/decam/DECam_CP -maxdepth 1 -type d`;do ln -s $fullnm $(basename $fullnm);done

Untar DR3 calibs:
a=fits_table('/scratch1/scratchdirs/desiproc/DRs/dr3-obiwan/legacypipe-dir/survey-ccds-dr3-eboss-ngc.fits.gz')
b=fits_table('/scratch1/scratchdirs/desiproc/DRs/dr3-obiwan/legacypipe-dir/survey-ccds-dr3-eboss-sgc.fits.gz')
a=np.array([num[:3] for num in a.expnum.astype(str)])
b=np.array([num[:3] for num in b.expnum.astype(str)])
c=set(a).union(set(b))
with open('calibs_2untar_eboss.txt','w') as foo:
    for num in list(set(c)):
        foo.write('%s\n' % num)         
foo.close()

dir=/global/project/projectdirs/cosmo/data/legacysurvey/dr3/calibs
for i in `head ../../../calibs_2untar_eboss.txt`;do echo $i; rsync -Lav ${dir}/psfex/legacysurvey_dr3_calib_decam_psfex_00${i}.tgz ./;done

MAKE 5 row survey-bricks table for decals_radeccolors.py
ngc=fits_table('survey-bricks-eboss-ngc.fits.gz')
b=fits_table('survey-bricks.fits.gz')


TEST before and after merge with master
In [6]: a.brickname[(a.nexp_g == 1)*(a.nexp_r == 1)*(a.nexp_z == 1)]
Out[6]: 
array(['1220p282', '1220p287', '1220p237', ..., '1723p260', '1724p202',
       '1724p200'],
Choose to test with brick 1220p282 
1) python obiwan/decals_sim_radeccolors.py --ra1 121.5 --ra2 122.5 --dec1 27.5 --dec2 28.7 --prefix finaltest --outdir /scratch2/scratchdirs/kaylanb/finaltest
2) for obj in star qso elg lrg;do sbatch submit_obiwan.sh $obj;done
e.g. 
export DECALS_SIM_DIR=/scratch2/scratchdirs/kaylanb/finaltest
python obiwan/decals_sim.py \
        --objtype $objtype --brick $brick --rowstart $rowstart \
        --add_sim_noise --prefix $prefix --threads $OMP_NUM_THREADS 
3) star,qso finished fine but elg,lrg ran out of memory
4) 

On desiproc:
export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/desiproc/DRs/dr3-obiwan/legacypipe-dir
export DECALS_SIM_DIR=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr3-obiwan/finaltest
python obiwan/decals_sim_radeccolors.py --ra1 121.5 --ra2 122.5 --dec1 27.5 --dec2 28.7 --prefix finaltest --outdir /scratch1/scratchdirs/desiproc/DRs/data-releases/dr3-obiwan/finaltest
python obiwan/decals_sim.py --objtype star --brick 1220p282 --rowstart 0 --add_sim_noise --prefix finaltest

TEST dr4:
export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/desiproc/DRs/dr4-bootes/legacypipe-dir
python legacypipe/runbrick.py --run dr4v2 --brick 1554p360 --skip --outdir test --nsigma 6

Tr yto hack decals_sim so we don't have to make copies of the data.

decals_sim -b 2428p117 -n 2000 --chunksize 500 -o STAR --seed 7618 --threads 15 > ~/2428p117.log 2>&1 & 

2428p117
3216p000
1315p240

nohup python ${LEGACYPIPE}/py/legacyanalysis/decals_sim.py -n 32 -c 10 -b 2428p117 -o STAR --zoom 500 1400 600 1300 > log2 2>&1 &
nohup python ${LEGACYPIPE}/py/legacyanalysis/decals_sim.py -n 5000 -c 500 -b 3216p000 -o STAR > & 3216p000.log &

import numpy as np
from legacypipe.survey import Decals, DecamImage, wcs_for_brick, ccds_touching_wcs

brickname = '2428p117'
decals = Decals()
brickinfo = decals.get_brick_by_name(brickname)
brickwcs = wcs_for_brick(brickinfo)
ccdinfo = decals.ccds_touching_wcs(brickwcs)
im = DecamImage(decals,ccdinfo[19])
targetrd = np.array([[ 242.98072831,   11.61900584],
                     [ 242.71332268,   11.61900584],
                     [ 242.71319548,   11.88093189],
                     [ 242.98085551,   11.88093189],
                     [ 242.98072831,   11.61900584]])
tim = im.get_tractor_image(const2psf=True, radecpoly=targetrd)

~1 target per square arcminute in DESI, so the random should have ~50 sources
per square arcminute: 

IDL> print, 50*3600.0*0.25^2
      11250.0 ; sources over a whole brick
"""
from __future__ import division, print_function

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import h5py
import galsim
import os
import sys
import shutil
import logging
import argparse
import pdb
import photutils

import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import resource_filename
from pickle import dump


from astropy.table import Table, Column, vstack
from astropy.io import fits
#from astropy import wcs as astropy_wcs
from fitsio import FITSHDR
import fitsio

from astropy import units
from astropy.coordinates import SkyCoord

from tractor.psfex import PsfEx, PsfExModel
from tractor.basics import GaussianMixtureEllipsePSF, RaDecPos

from legacypipe.runbrick import run_brick
from legacypipe.decam import DecamImage
from legacypipe.survey import LegacySurveyData, wcs_for_brick

import obiwan.decals_sim_priors as priors

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.ttime import Time

import csv

def write_dict(fn,d):
    '''d -- dictionary'''
    w = csv.writer(open(fn, "w"))
    for key, val in d.items():
        w.writerow([key, val])

def read_dict(fn):
    d = {}
    for key, val in csv.reader(open(fn)):
        d[key] = val
    return d

def imshow_stamp(stamp,fn='test.png',galsimobj=True):
    if galsimobj:
        img = stamp.array.copy()
    else:
        img= stamp.copy()
    img=img + abs(img.min())+1 
    plt.imsave(fn,np.log10(img),origin='lower',cmap='gray')
    #plt.imshow(np.log10(img),origin='lower',cmap='gray')
    #plt.savefig(fn)
    #plt.close()
    #print('Wrote %s' % fn)

def plot_radial_profs(fn,profs):
    assert(profs.shape[1] == 3)
    r=np.arange(profs.shape[0])
    for i,lab in zip(range(3),['src','srcnoise','srcnoiseimg']):
        plt.plot(r,profs[:,i],label=lab)
    plt.legend(loc='lower right')
    plt.savefig(fn)
    plt.close()


def ptime(text,t0):
    '''Timer'''    
    tnow=Time()
    print('TIMING:%s ' % text,tnow-t0)
    return tnow



def get_savedir(**kwargs):
    return os.path.join(kwargs['decals_sim_dir'],kwargs['objtype'],\
                        kwargs['brickname'][:3], kwargs['brickname'],\
                        "rowstart%d" % kwargs['rowst'])    

def get_fnsuffix(**kwargs):
    return '-{}-{}-{}.fits'.format(kwargs['objtype'], kwargs['brickname'],\
                                   'rowstart%d' % kwargs['rowst'])

class SimDecals(LegacySurveyData):
    def __init__(self, run=None, survey_dir=None, metacat=None, simcat=None, output_dir=None,\
                       add_sim_noise=False, folding_threshold=1.e-5, image_eq_model=False):
        '''folding_threshold -- make smaller to increase stamp_flux/input_flux'''
        super(SimDecals, self).__init__(survey_dir=survey_dir, output_dir=output_dir)
        self.run= run #None, eboss-ngc, eboss-sgc, etc
        self.metacat = metacat
        self.simcat = simcat
        # Additional options from command line
        self.add_sim_noise= add_sim_noise
        self.folding_threshold= folding_threshold
        self.image_eq_model= image_eq_model
        print('SimDecals: self.image_eq_model=',self.image_eq_model)

    def get_image_object(self, t):
        return SimImage(self, t)

    def ccds_for_fitting(survey, brick, ccds):
        return np.flatnonzero(ccds.camera == 'decam')

    def filter_ccds_files(self, fns):
        if self.run is None:
            return super(SimDecals,self).filter_ccds_files(fns)
        elif self.run == 'eboss-ngc':
            return [fn for fn in fns if
                 ('survey-ccds-dr3-eboss-ngc.fits.gz' in fn)]
        elif self.run == 'eboss-sgc':
            return [fn for fn in fns if
                 ('survey-ccds-dr3-eboss-sgc.fits.gz' in fn)]
        else:
            raise ValueError('run=%s not supported' % self.run)

def get_srcimg_invvar(stamp_ivar,img_ivar):
    '''stamp_ivar, img_ivar -- galsim Image objects'''
    # Use img_ivar when stamp_ivar == 0, both otherwise
    use_img_ivar= np.ones(img_ivar.array.shape).astype(bool)
    use_img_ivar[ stamp_ivar.array > 0 ] = False
    # First compute using both
    ivar= np.power(stamp_ivar.array.copy(), -1) + np.power(img_ivar.array.copy(), -1) 
    ivar= np.power(ivar,-1) 
    keep= np.ones(ivar.shape).astype(bool)
    keep[ (stamp_ivar.array > 0)*\
          (img_ivar.array > 0) ] = False
    ivar[keep] = 0.
    # Now use img_ivar only where need to
    ivar[ use_img_ivar ] = img_ivar.array.copy()[ use_img_ivar ]
    # return 
    obj_ivar = stamp_ivar.copy()
    obj_ivar.fill(0.)
    obj_ivar+= ivar
    return obj_ivar

def saturation_e(camera):
	# Saturation limit
	d=dict(decam=3e4) # e-
	return d[camera]

def ivar_to_var(ivar,nano2e=None,camera='decam'):
	assert(nano2e is not None)
	flag= ivar == 0.
	var= np.power(ivar, -1)
	# Set 0 ivar pixels to satuation limit
	# var * nano2e^2 = e-^2
	sat= saturation_e(camera) / nano2e**2
	var[flag]= sat
	return var 

class SimImage(DecamImage):
    def __init__(self, survey, t):
        super(SimImage, self).__init__(survey, t)
        self.t = t

    def get_tractor_image(self, **kwargs):
        tim = super(SimImage, self).get_tractor_image(**kwargs)
        if tim is None: # this can be None when the edge of a CCD overlaps
            return tim

        # Seed
        #if 'SEED' in self.survey.metacat.columns:
        #    seed = self.survey.metacat['SEED']
        #else:
        #    seed = None

        objtype = self.survey.metacat.get('objtype')[0]
        objstamp = BuildStamp(tim, gain=self.t.arawgain, \
                              folding_threshold=self.survey.folding_threshold,\
                              stamp_size= self.survey.metacat.stamp_size)

        # Grab the data and inverse variance images [nanomaggies!]
        tim_image = galsim.Image(tim.getImage())
        tim_invvar = galsim.Image(tim.getInvvar())
        tim_dq = galsim.Image(tim.dq)
        # Also store galaxy sims and sims invvar
        sims_image = tim_image.copy() 
        sims_image.fill(0.0)
        sims_ivar = sims_image.copy()
        # To make cutout for deeplearning
        tim.sims_xy = np.zeros((len(self.survey.simcat),4))-1 
        tim.sims_xyc = np.zeros((len(self.survey.simcat),2))-1
        tim.sims_id = np.zeros(len(self.survey.simcat)).astype(np.int32)-1
        tim.sims_added_flux = np.zeros(len(self.survey.simcat)).astype(float)-1

        # Store simulated galaxy images in tim object 
        # Loop on each object.
        for ii, obj in enumerate(self.survey.simcat):
            # Print timing
            t0= Time()
            if objtype in ['lrg','elg']:
                strin= 'Drawing 1 %s: sersicn=%.2f, rhalf=%.2f, ba=%.2f, phi=%.2f' % \
                        (objtype.upper(), obj.sersicn,obj.rhalf,obj.ba,obj.phi)
                print(strin)
            # Before drawing we can check if the obj is near CCD
            #if self.survey.metacat.cutouts[0]: 
            #    #draw_it= isNearCCD(tim,obj,
            #    junk,xx,yy = tim.wcs.wcs.radec2pixelxy(obj.ra,obj.dec)
            #    xx,yy= int(xx),int(yy)
            #    min_stamp_pixels= 16  # 200. / 3600. # arcsec -> deg
            #    obj_bounds= galsim.BoundsI(xmin= xx - min_stamp_pixels/2,\
            #                            xmax= xx + min_stamp_pixels/2,\
            #                            ymin= yy - min_stamp_pixels/2,\
            #                            ymax= yy + min_stamp_pixels/2)
            #    overlap = obj_bounds & tim_image.bounds
            #    # Even the SMALLEST stamp fits entirely within image
            #    # High prob teh full size stamp will  
            #    draw_it= obj_bounds == overlap
            #    #x1, y1 = tim.wcs.positionToPixel(RaDecPos(obj.ra-max_stamp_size/2, obj.dec-max_stamp_size/2))

            if objtype == 'star':
                stamp = objstamp.star(obj)
            elif objtype == 'elg':
                stamp = objstamp.elg(obj)
            elif objtype == 'lrg':
                stamp = objstamp.lrg(obj)
            elif objtype == 'qso':
                stamp = objstamp.qso(obj)
            t0= ptime('Drew the %s' % objtype.upper(),t0)
            #print('I predict we draw it',draw_it)
            # Save radial profiles after draw, addNoise, etc. for unit tests
            #rad_profs=np.zeros((stamp.array.shape[0],3))
            #rad_profs[:,0]= stamp.array.copy()[ stamp.array.shape[0]/2,: ]
            # Want to save flux actually added too
            added_flux= stamp.added_flux
            stamp_nonoise= stamp.copy()
            if self.survey.add_sim_noise:
                #stamp2,stamp3= objstamp.addGaussNoise(stamp, ivarstamp)
                ivarstamp= objstamp.addGaussNoise(stamp)
            # Add source if EVEN 1 pix falls on the CCD
            overlap = stamp.bounds & tim_image.bounds
            add_source = overlap.area() > 0
            # For Deep learning: only add source if entire thing fits on image
            if self.survey.metacat.cutouts[0]:
                # this is a deep learning run
                add_source= stamp.bounds == overlap
            if add_source:
                stamp = stamp[overlap]      
                ivarstamp = ivarstamp[overlap]      
                stamp_nonoise= stamp_nonoise[overlap]
                
                #rad_profs[:,1]= stamp.array.copy()[ stamp.array.shape[0]/2,: ]

                # Zero out invvar where bad pixel mask is flagged (> 0)
                keep = np.ones(tim_dq[overlap].array.shape)
                keep[ tim_dq[overlap].array > 0 ] = 0.
                ivarstamp *= keep
                #tim_invvar[overlap] *= keep # don't modify tim_invvar unless adding stamp ivar

                # Stamp ivar can get messed up at edges
                # especially when needed stamp smaller than args.stamp_size
                cent= ivarstamp.array.shape[0]/2
                med= np.median(ivarstamp.array[cent-2:cent+2,cent-2:cent+2].flatten() )
                # 100x median fainter gets majority of star,qso OR elg,lrg profile
                ivarstamp.array[ ivarstamp.array > 100 * med ] = 0.

                # Add stamp to image
                back= tim_image[overlap].copy()
                tim_image[overlap] = back.copy() + stamp.copy()
                # Add variances
                back_ivar= tim_invvar[overlap].copy()
                tot_ivar= get_srcimg_invvar(ivarstamp, back_ivar)
                tim_invvar[overlap] = tot_ivar.copy()

                #rad_profs[:,2]= tim_image[overlap].array.copy()[ stamp.array.shape[0]/2,: ]
                # Save sims info
                tim.sims_xy[ii, :] = [overlap.xmin-1, overlap.xmax-1,
                                      overlap.ymin-1, overlap.ymax-1] # galsim 1st index is 1
                tim.sims_xyc[ii, :] = [overlap.trueCenter().x-1, overlap.trueCenter().y-1]
                #tim.sims_radec[ii, :] = [obj.ra,obj.dec]
                tim.sims_id[ii] = obj.id
                tim.sims_added_flux[ii] = added_flux

                # For cutouts we only care about src, background, var (not ivar)
                if self.survey.metacat.cutouts[0]:
                    # Data for training: src+noise (cutout) and backgrn (cutout,var,badpix)
                    data= np.zeros((stamp.array.shape[0],stamp.array.shape[1],4))
					# FIX ME, add extra rotations for galaxies?
                    data[:,:,0]= stamp.array.copy() # src+noise
                    #data[:,:,1]= np.sqrt( np.power(stamp.array.copy(),2) )#src+noise var  #ivarstamp.array.copy() 
                    data[:,:,1]= back.array.copy() # back
                    data[:,:,2]= tim_dq[overlap].array.copy() # bad pix
                    data[:,:,3]= stamp_nonoise.array.copy() # Stamp w/out noise, sanity check
                    #data[:,:,2]= ivar_to_var(back_ivar.array.copy(),nano2e=objstamp.nano2e) # back var
                    #data[:,:,3]= tim_image[overlap].array.copy() # src+noise+background
                    #data[:,:,4]= tim_invvar[overlap].array.copy() # src+noise+background_ nvvar
                    # Save fn
                    brick= os.path.basename(os.path.dirname(self.survey.output_dir))
                    hdf5_fn= '%s_%s.hdf5' % (objtype,brick)  #'%s_%d_%s' % (tim.band,obj.id,expid)
                    hdf5_fn= os.path.join(self.survey.output_dir,hdf5_fn)
                    expid=str(tim.imobj).strip().replace(' ','')
                    node= '%s/%s/%s' % (obj.id,tim.band,expid)
                    fobj = h5py.File(hdf5_fn, "a")
                    dset = fobj.create_dataset(node, data=data,chunks=True)
                    for name,val,dtype in zip(\
                            ['id','flux_added'],\
                            [obj.id,added_flux],\
                            [np.int32,np.float32]):
                        dset.attrs.create(name,val,dtype=dtype)
                    #if objtype in ['lrg','elg']:
                    #    for name,val in zip(\
                    #            ['rhalf','sersicn','phi','ba'],\
                    #            [obj.rhalf,obj.sersicn,obj.phi,obj.ba]):
                    #        dset.attrs.create(name,val,dtype=np.float32)
                        #d.update(dict(rhalf=obj.rhalf,\
                        #              sersicn=obj.sersicn,\
                        #              phi=obj.phi,\
                        #              ba=obj.ba))
                    print('Saved %s to %s' % (node,hdf5_fn))
                    #np.save(fn+'.npy',data,allow_pickle=False)
                    # Save enough metadata to classify image quality later
                    #x1,x2,y1,y2= tuple(tim.sims_xy[ii,:])
                    #xc,yc= tuple(tim.sims_xyc[ii,:])
                    #d = dict(band=tim.band,\
					#		 expid=expid,\
                    #         addedflux= added_flux,\
                    #         id=obj.id,\
                    #         ra=obj.ra,\
                    #         dec=obj.dec)
                             #xc=xc,yc=yc,\
							 #(x1=x1,x2=x2,y1=y1,y2=y2,\
                             #gflux=obj.gflux,\
                             #rflux=obj.rflux,\
                             #zflux=obj.zflux)
                    #write_dict(fn+'.csv',d)
                    # Write sanity checks if they don't exists
                    #fns= glob(os.path.join(self.survey.output_dir,'*_src.fits'))
                    #if len(fns) == 0:
                    #    # Also write fits file for easier image stretching
                    #    fitsio.write(fn+'_src.fits',data[...,0],clobber=True)
                    #    fitsio.write(fn+'_src_invvar.fits',data[...,1],clobber=True)
                    #    fitsio.write(fn+'_img.fits',data[...,2],clobber=True)
                    #    fitsio.write(fn+'_img_invvar.fits',data[...,3],clobber=True)
                    #    fitsio.write(fn+'_srcimg.fits',data[...,4],clobber=True)
                    #    fitsio.write(fn+'_srcimg_invvar.fits',data[...,5],clobber=True)
                    #    # Draw Radial Profiles
                    #    plot_radial_profs(fn+'_profiles.png',rad_profs)
                
                #Extra
                sims_image[overlap] += stamp.copy() 
                sims_ivar[overlap] += ivarstamp.copy()
                
                    
                #print('HACK!!!')
                #galsim.fits.write(stamp, 'stamp-{:02d}.fits'.format(ii), clobber=True)
                #galsim.fits.write(ivarstamp, 'ivarstamp-{:02d}.fits'.format(ii), clobber=True)

                if np.min(sims_ivar.array) < 0:
                    log.warning('Negative invvar!')
                    import pdb ; pdb.set_trace()
        tim.sims_image = sims_image.array
        tim.sims_inverr = np.sqrt(sims_ivar.array)
        tim.sims_xy = tim.sims_xy.astype(int)
        tim.sims_xyc = tim.sims_xyc.astype(int)
        # Can set image=model, ivar=1/model for testing
        if self.survey.image_eq_model:
            tim.data = sims_image.array.copy()
            tim.inverr = np.zeros(tim.data.shape)
            tim.inverr[sims_image.array > 0.] = np.sqrt(1./sims_image.array.copy()[sims_image.array > 0.]) 
        else:
            tim.data = tim_image.array
            tim.inverr = np.sqrt(tim_invvar.array)
         
        #print('HACK!!!')
        #galsim.fits.write(invvar, 'invvar.fits'.format(ii), clobber=True)
        #import pdb ; pdb.set_trace()
        return tim

class BuildStamp():
    def __init__(self,tim, gain=4.0, folding_threshold=1.e-5, stamp_size=None):
        """Initialize the BuildStamp object with the CCD-level properties we need.
        stamp_size -- pixels, automatically set if None """
        self.band = tim.band.strip()
        self.stamp_size = stamp_size
        # GSParams should be used when galsim object is initialized
        # MAX size for sersic: 
        # https://github.com/GalSim-developers/GalSim/pull/450/commits/755bcfdca25afe42cccfd6a7f8660da5ecda2a65
        MAX_FFT_SIZE=1048576L #2^16=65536
        self.gsparams = galsim.GSParams(maximum_fft_size=MAX_FFT_SIZE,\
                                        folding_threshold=folding_threshold) 
        #print('FIX ME!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.gsdeviate = galsim.BaseDeviate()
        #if seed is None:
        #    self.gsdeviate = galsim.BaseDeviate()
        #else:
        #    self.gsdeviate = galsim.BaseDeviate(seed)

        self.wcs = tim.getWcs()
        self.psf = tim.getPsf()
        # Tractor wcs object -> galsim wcs object
        temp_hdr = FITSHDR()
        subwcs = tim.wcs.wcs.get_subimage(tim.wcs.x0, tim.wcs.y0,
                                  int(tim.wcs.wcs.get_width())-tim.wcs.x0,
                                  int(tim.wcs.wcs.get_height())-tim.wcs.y0)
        subwcs.add_to_header(temp_hdr)
        # Galsim uses astropy header, not fitsio
        hdr = fits.Header()
        for key in temp_hdr.keys(): hdr[key]=temp_hdr[key]
        self.galsim_wcs = galsim.GSFitsWCS(header=hdr)
        del subwcs,temp_hdr,hdr
        
        # zpscale equivalent to magzpt = self.t.ccdzpt+2.5*np.log10(self.t.exptime)
        self.zpscale = tim.zpscale      # nanomaggies-->ADU conversion factor
        self.nano2e = self.zpscale*gain # nanomaggies-->electrons conversion factor

    def setlocal(self,obj):
        """Get the pixel positions, local wcs, local PSF.""" 

        xx, yy = self.wcs.positionToPixel(RaDecPos(obj.get('ra'), obj.get('dec')))
        self.pos = galsim.PositionD(xx, yy)
        self.xpos = int(self.pos.x)
        self.ypos = int(self.pos.y)
        self.offset = galsim.PositionD(self.pos.x-self.xpos, self.pos.y-self.ypos)

        # galsim.drawImage() requires local (linear) wcs
        self.localwcs = self.galsim_wcs.local(image_pos=self.pos)
        #cd = self.wcs.cdAtPixel(self.pos.x, self.pos.y)
        #self.pixscale = np.sqrt(np.linalg.det(cd))*3600.0
        
        # Get the local PSF
        psfim = self.psf.getPointSourcePatch(self.xpos, self.ypos).getImage()
        #plt.imshow(psfim) ; plt.show()
        
        # make galsim PSF object
        self.localpsf = galsim.InterpolatedImage(galsim.Image(psfim), wcs=self.galsim_wcs,\
                                                 gsparams=self.gsparams)

    def addGaussNoise(self, stamp):
        """
        1) Adds gaussian noise to perfect source (in place)
        2) return invvar for the stamp
        Remember that STAMP and IVARSTAMP
        are in units of nanomaggies and 1/nanomaggies**2, respectively.
        """
        #stamp= stamp_backup.copy()
        #ivarstamp= ivarstamp_backup.copy()

        
        #varstamp = ivarstamp.copy()
        #ivarstamp.invertSelf() # input data, convert to variance
        #ivarstamp *= self.nano2e**2 # [electron^2]
             
        # Add the variance of the object to the variance image (in electrons).
        stamp *= self.nano2e       # [noiseless stamp, electron]
        stamp_var = galsim.Image(np.sqrt(stamp.array**2), wcs=self.galsim_wcs) 
        stamp_var.setOrigin(galsim.PositionI(stamp.xmin, stamp.ymin))

        # Add Poisson noise
        noise = galsim.VariableGaussianNoise(self.gsdeviate, stamp_var)
        #stamp2= stamp.copy()
        stamp.addNoise(noise)
        #stamp3= stamp2.copy()
        #c=np.random.normal(loc=0,scale=np.sqrt(objvar.array),size=objvar.array.shape)
        #noise = galsim.Image(c, wcs=self.galsim_wcs)
        #noise.setOrigin(galsim.PositionI(stamp.xmin, stamp.ymin))
        #stamp3+= noise
        
        # Variance of stamp+noise
        stamp_var = stamp.copy()
        stamp_var.fill(0.)
        stamp_var+= np.abs( stamp.array.copy() )
        
        #imshow_stamp(stamp,fn='std.png')
        #imshow_stamp(stamp_backup,'img.png')
        
        #b = galsim.Image(np.zeros(stamp.array.shape), wcs=self.galsim_wcs) 
        #b.array+= stamp.array.copy() 
        #b.array+= stamp_backup.array.copy() 
        #b= stamp.array.copy() + stamp_backup.array.copy()
        #b= stamp.copy()
        #b.drawImage(stamp_backup.copy(),add_to_image=True)
        #imshow_stamp(b,fn='std_img.png')
        # hists
        #for data,nam in zip([stamp.array.copy(),stamp_backup.array.copy(),b],['std','img','std_img']):
        #    j=plt.hist(data)
        #    plt.savefig(nam+'_hist.png')
        #    plt.close(nam+'_hist.png')
        # Convert back to [nanomaggies]
        stamp /= self.nano2e      
        #stamp2 /= self.nano2e      
        #stamp3 /= self.nano2e      
        stamp_var /= self.nano2e**2

        #ivarstamp = varstamp.copy()
        stamp_var.invertSelf()
        # Remask pixels that were masked in the original inverse variance stamp.
        #ivarstamp *= mask
        # This is now inv variance
        return stamp_var

    def convolve_and_draw(self,obj):
        """Convolve the object with the PSF and then draw it."""
        obj = galsim.Convolve([obj, self.localpsf], gsparams=self.gsparams)
        # drawImage() requires local wcs
        #try:
        if self.stamp_size is None:
            stamp = obj.drawImage(offset=self.offset, wcs=self.localwcs,method='no_pixel')
        else:
            stamp = obj.drawImage(offset=self.offset, wcs=self.localwcs,method='no_pixel',\
                                  nx=self.stamp_size,ny=self.stamp_size)
        
        #except SystemExit:
        #except BaseException:
        #    #logging.error(traceback.format_exc())
        #    print('got back drawImage!')
        #    raise ValueError
        #try: 
        #except:
        #    print("Unexpected error:", sys.exc_info()[0])
        #    raise
        stamp.setCenter(self.xpos, self.ypos)
        return stamp

    def star(self,obj):
        """Render a star (PSF)."""
        log = logging.getLogger('decals_sim')
        # Use input flux as the 7'' aperture flux
        self.setlocal(obj)
        psf = self.localpsf.withFlux(1.)
        if self.stamp_size is None:
            stamp = psf.drawImage(offset=self.offset, wcs=self.localwcs, method='no_pixel')
        else:
            stamp = psf.drawImage(offset=self.offset, wcs=self.localwcs, method='no_pixel',\
                                  nx=self.stamp_size,ny=self.stamp_size)
        # Fraction flux in 7'', FIXED pixelscale
        diam = 7/0.262
        # Aperture fits on stamp
        width= stamp.bounds.xmax-stamp.bounds.xmin
        height= stamp.bounds.ymax-stamp.bounds.ymin
        if diam > width and diam > height:
            nxy= int(diam)+2
            stamp = psf.drawImage(nx=nxy,ny=nxy, offset=self.offset, wcs=self.localwcs, method='no_pixel')
        assert(diam <= float(stamp.bounds.xmax-stamp.bounds.xmin))
        assert(diam <= float(stamp.bounds.ymax-stamp.bounds.ymin))
        # Aperture photometry
        apers= photutils.CircularAperture((stamp.trueCenter().x,stamp.trueCenter().y), r=diam/2)
        apy_table = photutils.aperture_photometry(stamp.array, apers)
        apflux= np.array(apy_table['aperture_sum'])[0]
        # Incrase flux so input flux contained in aperture
        flux = obj.get(self.band+'flux')*(2.-apflux/stamp.added_flux) # [nanomaggies]
        psf = self.localpsf.withFlux(flux)
        if self.stamp_size is None:
            stamp = psf.drawImage(offset=self.offset, wcs=self.localwcs, method='no_pixel')
        else:
            stamp = psf.drawImage(offset=self.offset, wcs=self.localwcs, method='no_pixel',\
                                  nx=self.stamp_size,ny=self.stamp_size)
        # stamp looses less than 0.01% of requested flux
        if stamp.added_flux/flux <= 0.9999:
            log.warning('stamp lost more than 0.01 percent of requested flux, stamp_flux/flux=%.7f',stamp.added_flux/flux)
        # test if obj[self.band+'FLUX'] really is in the 7'' aperture
        #apers= photutils.CircularAperture((stamp.trueCenter().x,stamp.trueCenter().y), r=diam/2)
        #apy_table = photutils.aperture_photometry(stamp.array, apers)
        #apflux= np.array(apy_table['aperture_sum'])[0]
        #print("7'' flux/input flux= ",apflux/obj[self.band+'FLUX'])
        
        # Convert stamp's center to its corresponding center on full tractor image
        stamp.setCenter(self.xpos, self.ypos)
        return stamp 

    def elg(self,obj):
        """Create an ELG (disk-like) galaxy."""
        # Create localpsf object
        self.setlocal(obj)
        objflux = obj.get(self.band+'flux') # [nanomaggies]
        try:
            galobj = galsim.Sersic(float(obj.get('sersicn')), half_light_radius=float(obj.get('rhalf')),\
                                flux=objflux, gsparams=self.gsparams)
        except:
            raise ValueError 
        galobj = galobj.shear(q=float(obj.get('ba')), beta=float(obj.get('phi'))*galsim.degrees)
        stamp = self.convolve_and_draw(galobj)
        return stamp

    def lrg(self,obj):
        """Create an LRG just like did for ELG"""
        return self.elg(obj)
    
    def qso(self,obj):
        """Create a QSO just like a star"""
        return self.star(obj)


#def no_overlapping_radec(ra,dec, bounds, random_state=None, dist=5.0/3600):
def no_overlapping_radec(Samp, dist=5./3600):
    '''Samp -- table containing id,ra,dec,colors
	returns bool indices to cut so not overlapping
    '''
    log = logging.getLogger('decals_sim')
    ra,dec= Samp.get('ra'),Samp.get('dec')
    #if random_state is None:
    #    random_state = np.random.RandomState()
    # ra,dec indices of just neighbors within "dist" away, just nerest neighbor of those
    cat1 = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
    cat2 = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
    i2, d2d, d3d = cat1.match_to_catalog_sky(cat2,nthneighbor=2) # don't match to self
    # Cut to large separations
    return np.array(d2d) > dist 

    #cnt = 1
    ##log.info("astrom: after iter=%d, have overlapping ra,dec %d/%d", cnt, len(m2),ra.shape[0])
    #log.info("Astrpy: after iter=%d, have overlapping ra,dec %d/%d", cnt, len(m2),ra.shape[0])
    #while len(m2) > 0:
    #    ra[m2]= random_state.uniform(bounds[0], bounds[1], len(m2))
    #    dec[m2]= random_state.uniform(bounds[2], bounds[3], len(m2))
    #    # Any more matches? 
    #    cat1 = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
    #    cat2 = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
    #    m2, d2d, d3d = cat1.match_to_catalog_sky(cat2,nthneighbor=2) # don't match to self
    #    b= np.array(d2d) <= dist
    #    m2= np.array(m2)[b]
    #    #
    #    cnt += 1
    #    log.info("after iter=%d, have overlapping ra,dec %d/%d", cnt, len(m2),ra.shape[0])
    #    if cnt > 30:
    #        log.error('Crash, could not get non-overlapping ra,dec in 30 iterations')
    #        raise ValueError
    #return ra, dec

#def build_simcat(nobj=None, brickname=None, brickwcs=None, meta=None, seed=None, noOverlap=True):
def build_simcat(Samp=None,brickwcs=None, meta=None):
    """Build the simulated object catalog, which depends on OBJTYPE."""
    log = logging.getLogger('decals_sim')

    #rand = np.random.RandomState(seed)

    # Assign central coordinates uniformly but remove simulated sources which
    # are too near to one another.  Iterate until we have the requisite number
    # of objects.
    #bounds = brickwcs.radec_bounds()
    #ra = rand.uniform(bounds[0], bounds[1], nobj)
    #dec = rand.uniform(bounds[2], bounds[3], nobj)
    #if noOverlap:
        #ra, dec= no_overlapping_radec(ra,dec, bounds,
        #                              random_state=rand,
        #                              dist=5./3600)
    # Cut to ra,dec that are sufficiently separated (> 5'' away from each other)
    I= no_overlapping_radec(Samp,dist=5./3600)
    skipping_ids= Samp.get('id')[I == False]
    Samp.cut(I)
    
    xxyy = brickwcs.radec2pixelxy(Samp.ra,Samp.dec)

    #cat = Table()
    #cat['ID'] = Column(Samp.get('id'),dtype='i4') #np.arange(nobj, dtype='i4'))
    #cat['RA'] = Column(Samp.ra, dtype='f8')
    #cat['DEC'] = Column(Samp.dec, dtype='f8')
    #cat['X'] = Column(xxyy[1][:], dtype='f4')
    #cat['Y'] = Column(xxyy[2][:], dtype='f4')
    cat = fits_table()
    for key in ['id','seed','ra','dec']:
        cat.set(key, Samp.get(key))
    cat.set('x', xxyy[1][:])
    cat.set('y', xxyy[2][:])

    typ=meta.get('objtype')[0]
    # Mags
    for key in ['g','r','z']:
        #print('WARNING: hardcoded mag = 19')
        #cat.set('%sflux' % key, 1E9*10**(-0.4* np.array([19.]*len(Samp)) ) ) # [nanomaggies]
        cat.set('%sflux' % key, 1E9*10**(-0.4*Samp.get('%s_%s' % (typ,key))) ) # [nanomaggies]
    # Galaxy Properties
    if typ in ['elg','lrg']:
        for key,tab_key in zip(['sersicn','rhalf','ba','phi'],['n','re','ba','pa']):
            cat.set(key, Samp.get('%s_%s'%(typ,tab_key) ))
        # Sersic n: GALSIM n = [0.3,6.2] for numerical stability,see
        # https://github.com/GalSim-developers/GalSim/issues/{325,450}
        # I'll use [0.4,6.1]
        vals= cat.sersicn
        vals[cat.sersicn < 0.4] = 0.4
        vals[cat.sersicn > 6.1] = 6.1
        cat.set('sersicn',vals)
        #cat['R50_1'] = Column(Samp.rhalf, dtype='f4')
        #cat['BA_1'] = Column(Samp.ba, dtype='f4')
        #cat['PHI_1'] = Column(Samp.phi, dtype='f4')
    return cat, skipping_ids



def get_parser():
    '''return parser object, tells it what options to look for
    options can come from a list of strings or command line'''
    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter,
                                     description='DECaLS simulations.')
    parser.add_argument('-o', '--objtype', type=str, choices=['star','elg', 'lrg', 'qso'], default='star', metavar='', help='insert these into images') 
    parser.add_argument('-b', '--brick', type=str, default='2428p117', metavar='', 
                        help='simulate objects in this brick')
    parser.add_argument('-rs', '--rowstart', type=int, default=0, metavar='', 
                        help='zero indexed, row of ra,dec,mags table, after it is cut to brick, to start on')
    parser.add_argument('--run', type=str, default=None, metavar='', 
                        help='tells which survey-ccds to read')
    parser.add_argument('--prefix', type=str, default='', metavar='', 
                        help='tells which input sample to use')
    parser.add_argument('-n', '--nobj', type=int, default=500, metavar='', 
                        help='number of objects to simulate (required input)')
    #parser.add_argument('-ic', '--ith_chunk', type=long, default=None, metavar='', 
    #                    help='run the ith chunk, 0-999')
    #parser.add_argument('-c', '--nchunk', type=long, default=1, metavar='', 
    #                    help='run chunks 0 to nchunk')
    parser.add_argument('-t', '--threads', type=int, default=1, metavar='', 
                        help='number of threads to use when calling The Tractor')
    #parser.add_argument('-s', '--seed', type=long, default=None, metavar='', 
    #                    help='random number seed, determines chunk seeds 0-999')
    parser.add_argument('-z', '--zoom', nargs=4, default=(0, 3600, 0, 3600), type=int, metavar='', 
                        help='see runbrick.py; (default is 0 3600 0 3600)')
    parser.add_argument('-survey-dir', '--survey_dir', metavar='', 
                        help='Location of survey-ccds*.fits.gz')
    #parser.add_argument('--rmag-range', nargs=2, type=float, default=(18, 26), metavar='', 
    #                    help='r-band magnitude range')
    parser.add_argument('--add_sim_noise', action="store_true", help="set to add noise to simulated sources")
    parser.add_argument('--folding_threshold', type=float,default=1.e-5,action="store", help="for galsim.GSParams")
    parser.add_argument('-testA','--image_eq_model', action="store_true", help="set to set image,inverr by model only (ignore real image,invvar)")
    parser.add_argument('--all-blobs', action='store_true', 
                        help='Process all the blobs, not just those that contain simulated sources.')
    parser.add_argument('--stage', choices=['tims', 'image_coadds', 'srcs', 'fitblobs', 'coadds'],
                        type=str, default='writecat', metavar='', help='Run up to the given stage')
    parser.add_argument('--early_coadds', action='store_true',
                        help='add this option to make the JPGs before detection/model fitting')
    parser.add_argument('--cutouts', action='store_true',
                        help='Stop after stage tims and save .npy cutouts of every simulated source')
    parser.add_argument('--stamp_size', type=int,action='store',default=64,\
                        help='Stamp/Cutout size in pixels')
    parser.add_argument('--bricklist',action='store',default='bricks-eboss-ngc.txt',\
                        help='if using mpi4py, $LEGACY_SURVEY_DIR/bricklist')
    parser.add_argument('--nproc', type=int,action='store',default=1,\
                        help='if using mpi4py')
    parser.add_argument('-v', '--verbose', action='store_true', help='toggle on verbose output')
    return parser
 
def create_metadata(kwargs=None):
    '''Parses input and returns dict containing 
    metatable,seeds,brickwcs, other goodies'''
    assert(kwargs is not None)
    log = logging.getLogger('decals_sim')
    # Pack the input parameters into a meta-data table and write out.
    #metacols = [
    #    ('BRICKNAME', 'S10'),
    #    ('OBJTYPE', 'S10'),
    #    ('NOBJ', 'i4'),
    #    ('CHUNKSIZE', 'i2'),
    #    ('NCHUNK', 'i2'),
    #    ('ZOOM', 'i4', (4,)),
    #    ('SEED', 'S20'),
    #    ('RMAG_RANGE', 'f4', (2,))]
    #metacat = Table(np.zeros(1, dtype=metacols))
    metacat = fits_table()
    for key in ['brickname','objtype']: #,'nchunk']:
        metacat.set(key, np.array( [kwargs[key]] ))
    metacat.set('nobj', np.array( [kwargs['args'].nobj] ))
    metacat.set('zoom', np.array( [kwargs['args'].zoom] ))
    metacat.set('cutouts', np.array( [kwargs['args'].cutouts] ))
    metacat.set('stamp_size', np.array( [kwargs['args'].stamp_size] ))
    #metacat['RMAG_RANGE'] = kwargs['args'].rmag_range
    #if not kwargs['args'].seed:
    #    log.info('Random seed = {}'.format(kwargs['args'].seed))
    #    metacat['SEED'] = kwargs['args'].seed
   
    #metacat_dir = os.path.join(kwargs['decals_sim_dir'], kwargs['objtype'],kwargs['brickname'][:3],kwargs['brickname'])    
    metacat_dir = get_savedir(**kwargs)
    if not os.path.exists(metacat_dir): 
        os.makedirs(metacat_dir)
    
    metafile = os.path.join(metacat_dir, 'metacat'+get_fnsuffix(**kwargs))
    log.info('Writing {}'.format(metafile))
    if os.path.isfile(metafile):
        os.remove(metafile)
    metacat.writeto(metafile)
    
    # Store new stuff
    kwargs['metacat']=metacat
    kwargs['metacat_dir']=metacat_dir


def create_ith_simcat(d=None):
    '''add simcat, simcat_dir to dict d
    simcat -- contains randomized ra,dec and PDF fluxes etc for ith chunk
    d -- dict returned by get_metadata_others()'''
    assert(d is not None)
    log = logging.getLogger('decals_sim')
    #chunksuffix = '{:02d}'.format(ith_chunk)
    # Build and write out the simulated object catalog.
    #seed= d['seeds'][ith_chunk]
    #simcat = build_simcat(d['nobj'], d['brickname'], d['brickwcs'], d['metacat'], seed)
    simcat, skipped_ids = build_simcat(Samp=d['Samp'],brickwcs=d['brickwcs'],meta=d['metacat'])
    # Simcat 
    simcat_dir = get_savedir(**d) #os.path.join(d['metacat_dir'],'row%d-%d' % (rowstart,rowend)) #'%3.3d' % ith_chunk)    
    if not os.path.exists(simcat_dir): 
        os.makedirs(simcat_dir)
    #simcatfile = os.path.join(simcat_dir, 'simcat-{}-{}-row{}-{}.fits'.format(d['brickname'], d['objtype'],rowstart,rowend)) # chunksuffix))
    simcatfile = os.path.join(simcat_dir, 'simcat'+get_fnsuffix(**d))
    if os.path.isfile(simcatfile):
        os.remove(simcatfile)
    simcat.writeto(simcatfile)
    log.info('Wrote {}'.format(simcatfile))
    # Skipped Ids
    if len(skipped_ids) > 0:
        skip_table= fits_table()
        skip_table.set('ids',skipped_ids)
        name= os.path.join(simcat_dir,'skippedids'+get_fnsuffix(**d))
        if os.path.exists(name):
            os.remove(name)
            log.info('Removed %s' % name)
        skip_table.writeto(name)
        log.info('Wrote {}'.format(name))
    # add to dict
    d['simcat']= simcat
    d['simcat_dir']= simcat_dir

def do_one_chunk(d=None):
    '''Run tractor
    Can be run as 1) a loop over nchunks or 2) for one chunk
    d -- dict returned by get_metadata_others() AND added to by get_ith_simcat()'''
    assert(d is not None)
    simdecals = SimDecals(run=d['run'],\
                          metacat=d['metacat'], simcat=d['simcat'], output_dir=d['simcat_dir'], \
                          add_sim_noise=d['args'].add_sim_noise, folding_threshold=d['args'].folding_threshold,\
                          image_eq_model=d['args'].image_eq_model)
    # Use Tractor to just process the blobs containing the simulated sources.
    if d['args'].all_blobs:
        blobxy = None
    else:
        blobxy = zip(d['simcat'].get('x'), d['simcat'].get('y'))

    # Format: run_brick(brick, survey obj, **kwargs)
    run_brick(d['brickname'], simdecals, threads=d['args'].threads, zoom=d['args'].zoom,
              wise=False, forceAll=True, hybridPsf=True, writePickles=False, do_calibs=True,
              write_metrics=False, pixPsf=True, blobxy=blobxy, early_coadds=d['args'].early_coadds,
              splinesky=True, ceres=False, stages=[ d['args'].stage ], plots=False,
              plotbase='sim',allbands='ugrizY')

def do_ith_cleanup(d=None):
    '''for each chunk that finishes running, 
    Remove unecessary files and give unique names to all others
    d -- dict returned by get_metadata_others() AND added to by get_ith_simcat()'''
    assert(d is not None) 
    log = logging.getLogger('decals_sim')
    log.info('Cleaning up...')
    brickname= d['brickname']
    output_dir= d['simcat_dir']
    shutil.copy(os.path.join(output_dir, 'tractor', brickname[:3],
                             'tractor-{}.fits'.format(brickname)),
                os.path.join(output_dir, 'tractor'+get_fnsuffix(**d)))
    for suffix in ('image', 'model', 'resid', 'simscoadd'):
        shutil.copy(os.path.join(output_dir,'coadd', brickname[:3], brickname,
                                 'legacysurvey-{}-{}.jpg'.format(brickname, suffix)),
                    os.path.join(output_dir, 'qa-'+suffix+ get_fnsuffix(**d).replace('.fits','.jpg')))
    shutil.rmtree(os.path.join(output_dir, 'coadd'))
    shutil.rmtree(os.path.join(output_dir, 'tractor'))
    log.info("Finished %s" % get_savedir(**d))

def get_sample_fn(brick,decals_sim_dir,prefix=''):
    fn= os.path.join(decals_sim_dir,'input_sample','bybrick','%ssample_%s.fits' % (prefix,brick))
    return fn
    #return os.path.join(decals_sim_dir,'softlinked_table') #'sample-merged.fits')

def main(args=None):
    """Main routine which parses the optional inputs."""
    t0= Time()
    # Command line options
    if hasattr(args,'__class__'):
        pass #args is probably an argparse.Namespace obj
    else:
        # Read a list of args from cmd line
        parser= get_parser()  
        args = parser.parse_args(args=args)
    
    if args.cutouts:
        args.stage = 'tims'
    # Setup loggers
    if args.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(level=lvl, stream=sys.stdout) #,format='%(message)s')
    log = logging.getLogger('decals_sim')
    # Sort through args 
    log.info('decals_sim.py args={}'.format(args))
    #max_nobj=500
    #max_nchunk=1000
    #if args.ith_chunk is not None: assert(args.ith_chunk <= max_nchunk-1)
    #assert(args.nchunk <= max_nchunk)
    #assert(args.nobj <= max_nobj)
    #if args.ith_chunk is not None: 
    #    assert(args.nchunk == 1) #if choose a chunk, only doing 1 chunk
    if args.nobj is None:
        parser.print_help()
        sys.exit(1)

    brickname = args.brick
    objtype = args.objtype
    maxobjs = args.nobj

    for obj in ('LSB'):
        if objtype == obj:
            log.warning('{} objtype not yet supported!'.format(objtype))
            return 0

    # Deal with the paths.
    if 'DECALS_SIM_DIR' in os.environ:
        decals_sim_dir = os.getenv('DECALS_SIM_DIR')
    else:
        decals_sim_dir = '.'
        
    #nchunk = args.nchunk
    #rand = np.random.RandomState(args.seed) # determines seed for all chunks
    #seeds = rand.random_integers(0,2**18, max_nchunk)

    log.info('Object type = {}'.format(objtype))
    #log.info('Number of objects = {}'.format(nobj))
    #log.info('Number of chunks = {}'.format(nchunk))

    # Optionally zoom into a portion of the brick
    survey = LegacySurveyData()
    brickinfo = survey.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brickinfo)
    W, H, pixscale = brickwcs.get_width(), brickwcs.get_height(), brickwcs.pixel_scale()

    log.info('Brick = {}'.format(brickname))
    if args.zoom is not None: # See also runbrick.stage_tims()
        (x0, x1, y0, y1) = args.zoom
        W = x1 - x0
        H = y1 - y0
        brickwcs = brickwcs.get_subimage(x0, y0, W, H)
        log.info('Zoom (pixel boundaries) = {}'.format(args.zoom))
    targetrd = np.array([brickwcs.pixelxy2radec(x, y) for x, y in
                         [(1,1), (W,1), (W,H), (1,H), (1,1)]])

    radec_center = brickwcs.radec_center()
    log.info('RA, Dec center = {}'.format(radec_center))
    log.info('Brick = {}'.format(brickname))
    t0= ptime('First part of Main()',t0)
    
    #if args.ith_chunk is not None: 
    #    chunk_list= [args.ith_chunk]
    #else: 
    #    chunk_list= range(nchunk)
    #chunk_list= [ int((args.rowstart)/maxobjs) ]

    # Ra,dec,mag table
    fn= get_sample_fn(brickname,decals_sim_dir,prefix=args.prefix)
    Samp= fits_table(fn)
    print('Reading input sample: %s' % fn)
    print('%d samples, for brick %s' % (len(Samp),brickname))
    # Already did these cuts in decals_sim_radeccolors 
    #r0,r1,d0,d1= brickwcs.radec_bounds()
    #Samp.cut( (Samp.ra >= r0)*(Samp.ra <= r1)*\
    #          (Samp.dec >= d0)*(Samp.dec <= d1) )
    # Sort by Sersic n low -> high (if elg or lrg)
    if objtype in ['elg','lrg']:
        if args.cutouts:
            # rhalf ~ 1-2'' at z ~ 1, n~1 
            #Samp=Samp[ (Samp.get('%s_re' % objtype) <= 10.)*\
            #           (Samp.get('%s_n' % objtype) <= 2.) ]
            Samp.set('%s_re' % objtype, np.array([1.5]*len(Samp)))
            Samp.set('%s_n' % objtype, np.array([1.]*len(Samp)))
        else:
            # Usual obiwan
            print('Sorting by sersic n')
            Samp=Samp[np.argsort( Samp.get('%s_n' % objtype) )]
        #    # Dont sort by sersic n for deeplearning cutouts
        #    print('NOT sorting by sersic n')
        #else:
    rowst,rowend= args.rowstart,args.rowstart+maxobjs
    if args.cutouts:
        # Gridded ra,dec for args.stamp_size x stamp_size postage stamps 
        size_arcsec= args.stamp_size * 0.262 * 2 #arcsec, 2 for added buffer
        # 20x20 grid
        dd= size_arcsec / 2. * np.arange(1,21,2).astype(float) #'' offsect from center
        dd= np.concatenate((-dd[::-1],dd))
        dd/= 3600. #arcsec -> deg
        # Don't exceed brick half width - 100''
        assert(dd.max() <= 0.25/2 - 100./3600)
        brickc_ra,brickc_dec= radec_center[0],radec_center[1]
        dec,ra = np.meshgrid(dd+ brickc_dec, dd+ brickc_ra) 
        dec= dec.flatten()
        ra= ra.flatten()
        assert(len(Samp) >= dec.size)
        keep= np.arange(dec.size)
        Samp.cut(keep)
        Samp.set('ra',ra)
        Samp.set('dec',dec)
    # Rowstart -> Rowend
    Samp= Samp[args.rowstart:args.rowstart+maxobjs]
    print('Max sample size=%d, actual sample size=%d' % (maxobjs,len(Samp)))
    assert(len(Samp) <= maxobjs)
    t0= ptime('Got input_sample',t0)

    # Store args in dict for easy func passing
    kwargs=dict(Samp=Samp,\
                brickname=brickname, \
                decals_sim_dir= decals_sim_dir,\
                brickwcs= brickwcs, \
                objtype=objtype,\
                nobj=len(Samp),\
                maxobjs=maxobjs,\
                rowst=rowst,\
                rowend=rowend,\
                args=args)

    # Stop if starting row exceeds length of radec,color table
    if len(Samp) == 0:
        fn= get_savedir(**kwargs)+'_exceeded.txt'
        junk= os.system('touch %s' % fn)
        print('Wrote %s' % fn)
        raise ValueError('starting row=%d exceeds number of artificial sources, quit' % rowst)
    
    # Create simulated catalogues and run Tractor
    create_metadata(kwargs=kwargs)
    t0= ptime('create_metadata',t0)
    # do chunks
    #for ith_chunk in chunk_list:
    #log.info('Working on chunk {:02d}/{:02d}'.format(ith_chunk,kwargs['nchunk']-1))
    # Random ra,dec and source properties
    create_ith_simcat(d=kwargs)
    t0= ptime('create_ith_simcat',t0)
    # Run tractor
    kwargs.update( dict(run=args.run) )
    do_one_chunk(d=kwargs)
    t0= ptime('do_one_chunk',t0)
    # Clean up output
    if args.cutouts == False:
        do_ith_cleanup(d=kwargs)
        t0= ptime('do_ith_cleanup',t0)
    log.info('All done!')
     
if __name__ == '__main__':
    main()
