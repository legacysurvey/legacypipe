import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from glob import glob

from astrometry.util.fits import fits_table

class Translator(object):
    def __init__(self,j,a,verbose=True,savefig=True):
        self.j=j
        self.a=a
        self.verbose=verbose
        self.savefig=savefig
        #
        sj,sa= set(self.j.get_columns()),set(self.a.get_columns())
        self.both=list(sj.intersection(sa))
        self.a_only=list(sa.difference(sj))
        self.j_only=list(sj.difference(sa))
#         self.print_set('both',self.both)
#         self.print_set('a_only',self.a_only)
#         self.print_set('j_only',self.j_only)
        #
        self.translate()
        # numeric vs. str keys
        self.float_or_str()
        # What did we miss?
        if self.verbose:
            print '----\nmissing from Johns\n----'
            for key in j.get_columns():
                if key not in self.j2a.keys():
                    print '%s' % key
            vals=[v for k,v in self.j2a.items()]
            print '----\nmissing from Arjuns\n----'
            for key in a.get_columns():
                if key not in vals:
                    print '%s' % key
        # Problematic keys
        self.discrep_keys=['zpt','zptavg',\
                           'skycounts','skyrms','skymag',\
                           'transp','phoff','rarms','decrms',\
                           'raoff','decoff']
                
    def compare(self):
        # Comparisons
        self.compare_strings()
        self.compare_floats_ints()
            
        
    def map_a2j(self,key):
        d= dict(ra='ra_bore',dec='dec_bore',\
                arawgain='gain',\
                ccdhdunum='image_hdu',\
                zpt='zptavg',\
                filename='image_filename',\
                naxis1='width',\
                naxis2='height')
        return d[key]
        
    def translate(self):
        j2a={}
        skip=[]
        for key in self.both:
            if key in ['zpt','ra','dec']:
                skip+= [key]
            else:
                j2a[key]=key
        for key in self.a_only:
            if np.any((key.startswith('ccdhdu'),\
                       key.startswith('ccdnmatcha'),\
                       key.startswith('ccdnmatchb'),\
                       key.startswith('ccdnmatchc'),\
                       key.startswith('ccdnmatchd'),\
                       key.startswith('ccdzpta'),\
                       key.startswith('ccdzptb'),\
                       key.startswith('ccdzptc'),\
                       key.startswith('ccdzptd'),\
                       key.startswith('ccdnum')),axis=0):
                skip+= [key]
            elif key.startswith('ccd'):
                j2a[key.replace('ccd','')]=key
            else:
                skip+= [key]
        for key in skip:
            try: 
                j2a[ self.map_a2j(key) ]= key
            except KeyError:
                pass # Missing this key, will find these later
        self.j2a=j2a
        if self.verbose:
            print 'John --> Arjun'
            for key in self.j2a.keys():
                print '%s --> %s' % (key,self.j2a[key])
    
    def float_or_str(self):
        self.typ=dict(floats=[],ints=[],strs=[])
        for key in self.j2a.keys():
            typ= type(self.j.get(key)[0])
            if np.any((typ == np.float32,\
                       typ == np.float64),axis=0):
                self.typ['floats']+= [key]
            elif np.any((typ == np.int16,\
                         typ == np.int32),axis=0):
                self.typ['ints']+= [key]
            elif typ == np.string_:
                self.typ['strs']+= [key]
            else:
                print 'WARNING: unknown type for key=%s, ' % key,typ
    
    def print_set(self,text,s):
        print '%s\n' % text,np.sort(list(s))
        
    def compare_strings(self):
        print '----\nString comparison, John --> Arjun\n----'
        for s_key in self.typ['strs']:
            print '%s:%s --> %s:%s' % (s_key, self.j.get(s_key)[0],\
                                       self.j2a[s_key], self.a.get( self.j2a[s_key] )[0])
            
    def compare_floats_ints(self):
        panels=len(self.typ['floats'])+len(self.typ['ints'])
        cols=3
        if panels % cols == 0:
            rows=panels/cols
        else:
            rows=panels/cols+1
        # print cols,rows
        fig,axes= plt.subplots(rows,cols,figsize=(20,30))
        ax=axes.flatten()
        plt.subplots_adjust(hspace=0.4,wspace=0.3)
        cnt=-1
        for key in self.typ['floats']+self.typ['ints']:
            cnt+=1
            arjun= self.a.get( self.j2a[key] )
            john= self.j.get(key)
            #if key in ['transp','raoff','decoff','rarms','decrms',\
            #           'phrms','phoff','skyrms','skycounts',\
            #           'nstar','nmatch','width','height','mdncol']:
            ax[cnt].scatter(arjun,(john-arjun)/arjun) 
            ylab=ax[cnt].set_ylabel('(John-Arjun)/Arjun',fontsize='small')
            xlab=ax[cnt].set_xlabel('%s (Arjun)' % self.j2a[key],fontsize='small')
            #y= (arjun-john)/john
            #ax[cnt].scatter(john,y) 
            #ylab=ax[cnt].set_ylabel('(Arjun-John)/John',fontsize='small')
            #ax[cnt].set_ylim([-0.1,0.1])
        if self.savefig:
            fn="float_comparison.png"
            plt.savefig(fn,\
                        bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            print('Wrote %s' % fn)
 
    def compare_problematic(self):
        print('comparing problematic')
        # Compare
        panels=len(self.discrep_keys)
        cols=3
        if panels % cols == 0:
            rows=panels/cols
        else:
            rows=panels/cols+1
        fig,axes= plt.subplots(rows,cols,figsize=(20,10))
        ax=axes.flatten()
        plt.subplots_adjust(hspace=0.4,wspace=0.3)
        cnt=-1
        for key in self.discrep_keys:
            cnt+=1
            arjun= self.a.get( self.j2a[key] )
            john= self.j.get(key)
            #if key == 'skymag':
            #    arjun=arjun[ self.j.get('exptime') > 50]
            #    john=john[ self.j.get('exptime') > 50]
            ax[cnt].scatter(john,arjun)
            ylab=ax[cnt].set_ylabel('%s (Arjun)' % self.j2a[key],fontsize='large')
            xlab=ax[cnt].set_xlabel('%s (John)' % key,fontsize='large')
        if self.savefig:
            fn="problemkeys_comparison.png"
            plt.savefig(fn,\
                        bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            print('Wrote %s' % fn)
       
    def save(self,savefn='translate.pickle'):
        if hasattr(self,'j2a'):
            self.savefn= savefn
            fout=open(self.savefn,'w')
            pickle.dump(self.j2a,fout)
            fout.close()
            print('Wrote %s' % self.savefn)
        else: 
            raise ValueError('not translation dictionary: self.j2a')
    
    def restore(self):
        if hasattr(self,'savefn'):
            fin=open(fn,'r')
            my_dict= pickle.load(fin)
            fin.close()
            return my_dict
        else: 
            raise ValueError('file has not been written to be restored')
    

class Converter(object):
    def __init__(self,j_to_convert,a):
        self.indir='/global/homes/k/kaylanb/repos/thesis_code/zeropoints/data/original_backup'
        # Build translation with existing data
        j_fixed= fits_table(os.path.join(self.indir,'zeropoint-k4m_160203_015632_ooi_zd_v2.fits'))
        a_fixed= fits_table(os.path.join(self.indir,'arjun_zeropoint-k4m_160203_015632_ooi_zd_v2.fits'))
        self.trans=Translator(j_fixed,a_fixed,verbose=False)
        # Convert it to something that matches Arjuns
        self.plot_converted(j_to_convert,a)
        
    def convert_j2a(self,j,key):
        if key in ['skycounts','skyrms']:
            return j.get(key)/j.get('exptime')
        elif key in ['skymag']:
            return j.get(key)-2.5*np.log10(j.get('gain'))
        else:
            return j.get(key)

    def plot_converted(self,j_to_convert,a):
        print('converted, now comparing')
        panels=len(self.trans.discrep_keys)
        cols=3
        if panels % cols == 0:
            rows=panels/cols
        else:
            rows=panels/cols+1
        fig,axes= plt.subplots(rows,cols,figsize=(20,10))
        ax=axes.flatten()
        plt.subplots_adjust(hspace=0.4,wspace=0.3)
        cnt=-1
        for key in self.trans.discrep_keys:
            cnt+=1
            arjun= a.get( self.trans.j2a[key] )
            john= self.convert_j2a(j_to_convert, key)
            if key == 'skymag':
                arjun=arjun[ j_to_convert.get('exptime') > 50]
                john=john[ j_to_convert.get('exptime') > 50]
            ax[cnt].scatter(john,arjun)
            ylab=ax[cnt].set_ylabel('%s (Arjun)' % self.trans.j2a[key],fontsize='large')
            xlab=ax[cnt].set_xlabel('%s (John)' % key,fontsize='large')
            
    def rename_john_zptfile(self,fn=None):
        assert(fn is not None)
#         mydir='/global/homes/k/kaylanb/repos/thesis_code/zeropoints/data'
#         fn= os.path.join(mydir,'zeropoint-k4m_160203_015632_ooi_zd_v2.fits')
        j= fits_table(fn)
        jcopy= fits_table(fn)
        # Remove johns keys
        for key in self.trans.j2a.keys():
            j.delete_column(key)
        # Replace with arjun's keys, but John's data for that key
        for key in self.trans.j2a.keys():
            j.set(self.trans.j2a[key], jcopy.get(key))
        # Convert quantities to Arjuns based on empirical corrections 
        for key in ['skycounts','skyrms','skymag']:
            print('BEFORE: key=%s,j.get(key)[:4]=' % self.trans.j2a[key],j.get(self.trans.j2a[key])[:4])
            j.set(self.trans.j2a[key], self.convert_j2a(jcopy,key))
            print('AFTERE: key=%s,j.get(key)[:4]=' % self.trans.j2a[key],j.get(self.trans.j2a[key])[:4])
        # Save
        name= fn.replace('.fits','_renamed.fits')
        if os.path.exists(name):
            os.remove(name)
        j.writeto(name)
        print('Wrote renamed file: %s' % name)
    
    def compare_problematic(self):
        # Assumes j results from self.rename_john_zptfile() 
        j=fits_table('/project/projectdirs/desi/users/burleigh/test_data/old/john_combined_renamed.fits')
        a=fits_table('/project/projectdirs/desi/users/burleigh/test_data/old/arjun_combined.fits')
        # Compare
        panels=len(discrep_keys)
        cols=3
        if panels % cols == 0:
            rows=panels/cols
        else:
            rows=panels/cols+1
        fig,axes= plt.subplots(rows,cols,figsize=(20,10))
        ax=axes.flatten()
        plt.subplots_adjust(hspace=0.4,wspace=0.3)
        cnt=-1
        for key in self.trans.discrep_keys:
            cnt+=1
            arjun= a.get( self.trans.j2a[key] )
            john= j.get( self.trans.j2a[key] )
            if key == 'skymag':
                arjun=arjun[ j.get('exptime') > 50]
                john=john[ j.get('exptime') > 50]
            ax[cnt].scatter(john,arjun)
            ylab=ax[cnt].set_ylabel('Arjun',fontsize='large')
            xlab=ax[cnt].set_xlabel('%s (John)' % key,fontsize='large')

def main(**kwargs):
    import theValidator.catalogues as cats
    camera= kwargs.get('camera','mosaic')
    savefig= kwargs.get('savefig',True)
    original= kwargs.get('original',False)
    project= kwargs.get('project',False)
 
    # Read in files
    if original:
        mydir='/global/homes/k/kaylanb/repos/thesis_code/zeropoints/data/'+camera
        print('basedir= %s' % mydir)
        if camera == 'mosaic':
            # part='k4m_160203_015632_ooi_zd_v2'
            part='k4m_160203_064552_ooi_zd_v2'
            j_stars=fits_table(os.path.join(mydir,'stars-gain1.8noCorr/','stars-gain1.8noCorrzeropoint-'+part+'-stars.fits'))
            a_stars=fits_table(os.path.join(mydir,'original/','matches-'+part+'.fits'))
            j=fits_table(os.path.join(mydir,'stars-gain1.8noCorr/','stars-gain1.8noCorrzeropoint-'+part+'.fits'))
            j_mzls_color= fits_table(os.path.join(mydir,'stars-gain1.8noCorr/','stars-gain1.8noCorrmzlscolorzeropoint-'+part+'.fits'))
            a=fits_table(os.path.join(mydir,'original/','arjun_zeropoint-'+part+'.fits'))
        elif camera=='90prime':
            suffix='ksb_160211_123455_ooi_g_v1.fits'
            a=fits_table(os.path.join(mydir,'arjun','zeropoint-'+suffix))
            a_stars=fits_table(os.path.join(mydir,'arjun','matches-'+suffix))
            j=fits_table(os.path.join(mydir,'kay1/','blahzeropoint-'+suffix))
            j_stars=fits_table(os.path.join(mydir,'kay1/','blahzeropoint-'+suffix.replace('.fits','-stars.fits')))
    else:
        # Stack and compare newest
        if camera == 'mosaic':
            aka='mzls'
            mydir='/scratch2/scratchdirs/kaylanb/cosmo/staging/mosaicz/MZLS_CP/CP20160202v2/zpts'
            if project:
                mydir='/global/projecta/projectdirs/cosmo/work/dr4/zpt/mosaic'
            zpt_fns= glob(os.path.join(mydir,'mzlszeropoint-*v2.fits'))
            star_fns= glob(os.path.join(mydir,'mzlszeropoint-*v2-stars.fits'))
            # Arjuns
            a=fits_table('~arjundey/ZeroPoints/mzls-v2-zpt-all-2016dec06.fits')
        elif camera == '90prime':
            aka='90prime'
            mydir='/scratch2/scratchdirs/kaylanb/cosmo/staging/bok/BOK_CP/CP20160102/zpts'
            if project:
                mydir='/global/projecta/projectdirs/cosmo/work/dr4/zpt/90prime'
            zpt_fns= glob(os.path.join(mydir,'90primezeropoint-*v1.fits'))
            star_fns= glob(os.path.join(mydir,'90primezeropoint-*v1-stars.fits'))
            # Arjuns
            a=fits_table('~arjundey/ZeroPoints/bass-zpt-all-2016dec06.fits')
        # General to either camera
        assert(len(zpt_fns) > 0 and len(star_fns) > 0)
        j= cats.CatalogueFuncs().stack(zpt_fns,textfile=False)    
        #j_stars= cats.CatalogueFuncs().stack(star_fns,textfile=False)   
        # Match Arjun's ALL zpt file to these frames
        keep=np.zeros(len(a)).astype(bool)
        for fn in j.image_filename:
            keep[ np.where(a.filename == fn.replace('.fz',''))[0] ]=True
        a.cut(keep)
        # Same order: filename && ccdname
        a=a[ np.argsort( np.char.replace(a.filename,"ooi",np.char.lower(a.ccdname)) ) ]
        j=j[ np.argsort( np.char.replace(j.image_filename,"ooi",np.char.lower(j.ccdname)) ) ]
        print('ARE THESES IN SAME ORDE?')
        print np.char.lower(a.ccdname)[:12]
        print np.char.lower(j.ccdname)[:12]
        #j= j[np.argsort(j.image_filename)]
        #a= a[np.argsort(a.filename)]
        # Quality Cuts
        keep= np.zeros(len(j)).astype(bool)
        if camera == 'mosaic':
            keep[ (a.exptime > 40.)*(a.ccdnmatch > 50)*(a.ccdzpt > 25.8)*\
                  (j.exptime > 40.)*(j.nmatch > 50)*   (j.zpt > 25.8) ]=True
        elif camera == '90prime':
            keep[ (a.ccdzpt >= 20.)*(a.ccdzpt <= 30.)*\
                  (j.zpt >= 20.)*   (j.zpt <= 30.) ]=True
        a.cut(keep)
        j.cut(keep)
        assert(len(a) == len(j))
    # Match and compare
    #for key in ['ra','dec']: 
    #    a_stars.set(key,a_stars.get('ccd_%s' % key))
    #imatch,imiss,d2d= cats.Matcher().match_within(a_stars,j_stars)
    trans=Translator(j,a,verbose=True)
    trans.compare()
    trans.compare_problematic()
    return a,j,trans

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.') 
    parser.add_argument('--camera',choices=['90prime','mosaic'],type=str,action='store',required=True) 
    parser.add_argument('--project',action='store_true',default=False,help='load data from /project not /scratch',required=False) 
    parser.add_argument('--original',action='store_true',default=False,help='do original comparison where reproduce arjuns zpts except for radec offsets',required=False) 
    parser.add_argument('--savefig',action='store_true',default=False,required=False) 
    args = parser.parse_args()

    kwargs=dict(camera=args.camera,original=args.original,\
                savefig=args.savefig)
    a,j,trans=main(**kwargs)
    raise ValueError


#    j_cat,a_cat= [],[]
#    # mydir='/project/projectdirs/desi/users/burleigh/test_data/old'
#    mydir='/global/homes/k/kaylanb/repos/thesis_code/zeropoints/data/'
#    fns=glob.glob(os.path.join(mydir,'original/originalzero*v2.fits'))
#    for fn in fns:
#        j_cat.append( fits_table(fn) )
#        a_fn= os.path.join(os.path.dirname(fn), 'arjun_'+os.path.basename(fn).replace('original',''))
#        a_cat.append( fits_table(a_fn) )
#    # fns=glob.glob('/project/projectdirs/desi/users/burleigh/test_data/zeropoint*v2.fits')
#    # for fn in fns:
#    #     a_cat.append( fits_table(fn) )
#    #     j_fn= os.path.join(os.path.dirname(fn), 'units_1arcsec_'+os.path.basename(fn))
#    #     j_cat.append( fits_table(j_fn) )
#    j = merge_tables(j_cat, columns='fillzero')
#    a = merge_tables(a_cat, columns='fillzero')
#
#    t=Translator(j,a,verbose=False)
#    t.compare_problematic()
#
#    c=Converter(j,a)    

