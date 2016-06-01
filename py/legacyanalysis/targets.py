import numpy as np
from astropy.io import fits
from astrometry.util.fits import fits_table

def tractor_cat(fn):
    '''uses Dustin's fits_table() function to read his tractor catalog then returns it as dictionary of np arrays
    his fits_table handles booleans with weird values like 84 for "brick_primary", and converts that to True/False'''
    temp = fits_table(fn)
    data={}
    for key in temp.columns(): data[key]= temp.get(key)
    return data

def read_from_tractor_cat(fn,same_keys=['ra','dec','type']):
	'''returns data dict for DECaLS()
	same_keys -- keys that have identical names b/w tractor cat and data dict expected by DeCALS()'''
	a=tractor_cat(fn)
	data={}
	for k in same_keys:
		data[k]= a[k]
	data['type']= np.char.strip(data['type'])
	for band,ind in zip(['g','r','z'],[1,2,4]):
		data[band+'flux']= a['decam_flux'][:,ind]
		data[band+'flux_ivar']= a['decam_flux_ivar'][:,ind]
		data[band+'_ext']= a['decam_mw_transmission'][:,ind]
		data[band+'_psf_fwhm']= a['decam_psfsize'][:,ind]
	for band,ind in zip(['w1'],[0]):
		data[band+'flux']= a['wise_flux'][:,ind]
		data[band+'flux_ivar']= a['wise_flux_ivar'][:,ind]
		data[band+'_ext']= a['wise_mw_transmission'][:,ind]
	return data

def read_from_psql_file(fn,use_cols=range(14),str_cols=['type']):
	'''return data dict for DECaLS()
	fn -- file name of psql db txt file
	use_cols -- list of column indices to get, first column is 0
	str_cols -- list of column names that should have type str not float'''
	#get column names
	fin=open(fn,'r')
	cols=fin.readline()
	fin.close()
	cols= np.char.strip( np.array(cols.split('|'))[use_cols] )
	#get data
	arr=np.loadtxt(fn,dtype='str',comments='(',delimiter='|',skiprows=2,usecols=use_cols)
	data={}
	for i,col in enumerate(cols):
		if col in str_cols: data[col]= np.char.strip( arr[:,i].astype(str) )
		else: data[col]= arr[:,i].astype(float)
	return data

def data_extract(data,indices):
    '''data -- returned by read_from_*
    indicies -- return data but only at these indices'''
    #return {data[k][indices] for 
    cpy={}
    for k in data.keys(): 
        cpy[k]= data[k][indices]
    return cpy


class PTF(object):
	def __init__(self,data): #,cut_neg_flux=False):
		#data from psql db txt file
		self.data= data
		self.bands=['g','r']
		#convert to AB mag and select targets
		#if cut_neg_flux: self.cut_neg_fluxes()
		self.flux_w_ext()
		build_nan_inf_mask(self.data,self.bands) #mask out, don't remove neg. fluxes
		self.flux_to_mag_ab()
	
	#def cut_neg_fluxes(self):
	#	ind= np.all((self.data['gflux']>0,self.data['rflux']>0), axis=0)
	#	for key in self.data.keys(): self.data[key]= self.data[key][ind]

	def flux_w_ext(self):
		for b in ['g', 'r']:
			self.data[b+'flux_ext']= self.data[b+'flux']/self.data[b+'_ext']

	def flux_to_mag_ab(self):
		for b in ['g', 'r']:
			self.data[b+'mag']= -2.5*np.log10(self.data[b+'flux_ext'])	


class DECaLS(object):
    def __init__(self,data,w1=False,w2=False): #,cut_neg_flux=False):
        #store data, data from psql db txt file or tractor cat
        self.data= data
        self.bands= ['g', 'r', 'z']
        if w1: self.bands+= ['w1']
        if w2: self.bands+= ['w2']
        #create,apply masks
        self.create_and_apply_masks() 
        #calculate everything using masks
        self.calc_everything()
        #update masks for data and everything calc from data
        #self.update_masks_for_everything(mask=None,mask_wise=None)
        print "DECaLS object initialized"

    def create_and_apply_masks(self):
        '''create mask and mask_wise, apply them to all self.data.keys() fields'''
        #PRIMARY: mask is ANY of the following conditions are true
        self.mask=np.any((self.data['gflux'] <= 0,\
                        self.data['rflux'] <= 0,\
                        self.data['zflux'] <= 0),axis=0)
        #SECONDARY: mask + where wise fluxes < 0, only used when use wise fluxes for something, eg lrg selection
        if 'w1' in self.bands and 'w2' in self.bands:
            self.mask_wise= np.any((self.mask,\
                                    self.data['w1flux'] <= 0,\
                                    self.data['w1flux'] <= 0),axis=0)
        elif 'w1' in self.bands: 
            self.mask_wise= np.any((self.mask,\
                                    self.data['w1flux'] <= 0),axis=0)
        else: self.mask_wise= self.mask
        #propogate masks through data
        for key in self.data.keys():
            if key.startswith('w1') or key.startswith('w2'):
                self.data[key]= np.ma.masked_array(self.data[key],mask=self.mask_wise)
            else: 
                self.data[key]= np.ma.masked_array(self.data[key],mask=self.mask)
    #target selection
    #np.all() on a masked array does NOT return a masked array, require both target and good object
    def BGS_cuts(self):
        i_bgs= np.all((self.data['type'] != 'PSF',\
                       self.data['rmag']<19.35),axis=0)
        good= self.mask == False
        self.bgs= np.all((i_bgs,good),axis=0) #both target and good object
    def LRG_cuts(self):
        if 'w1' in self.bands:
            i_lrg= np.all((self.data['rmag']<23.0,\
                                self.data['zmag']<20.56,\
                                self.data['w1mag']<19.35,\
                                self.data['rmag']-self.data['zmag']> 1.6,\
                                self.data['rmag']-self.data['w1mag']> 1.33*(self.data['rmag']-self.data['zmag']) -0.33),axis=0)
        else:
            i_lrg= np.zeros(self.data['rmag'].size,dtype=bool) #python 0 = False
            print "WARNING: no LRGs selected because no W1"
        good= self.mask_wise == False
        self.lrg= np.all((i_lrg,good),axis=0) #both target and good object

    def ELG_cuts(self):
        i_elg= np.all((self.data['rmag']<23.4,\
                    self.data['rmag']-self.data['zmag']> 0.3,\
                    self.data['rmag']-self.data['zmag']< 1.5,\
                    self.data['gmag']-self.data['rmag']< 1.0*(self.data['rmag']-self.data['zmag']) -0.2,\
                    self.data['gmag']-self.data['rmag']< -1.0*(self.data['rmag']-self.data['zmag']) +1.2),axis=0)
        good= self.mask == False
        self.elg= np.all((i_elg,good),axis=0) #both target and good object

    def QSO_cuts(self):
        if 'w1' in self.bands and 'w2' in self.bands:
            wavg= 0.75*self.data['w1mag']+ 0.25*self.data['w2mag']
            i_qso= np.all((self.data['type']=='PSF ',\
                            self.data['rmag']<23.0,\
                            self.data['gmag']-self.data['rmag']< 1.0,\
                            self.data['rmag']-self.data['zmag']> -0.3,\
                            self.data['rmag']-self.data['zmag']< 1.1,\
                            self.data['rmag']-wavg> 1.2*(self.data['gmag']-self.data['rmag']) -0.4),axis=0)
        else: 
            i_qso= np.zeros(self.data['rmag'].size,dtype=bool) #python 0 = False
            print "WARNING: no QSOs selected because no W1,W2"
        good= self.mask_wise == False
        self.qso= np.all((i_qso,good),axis=0) #both target and good object

    def calc_everything(self):
        '''compute flux/ext, mags, etc., vals automatically masked where self.mask,self.mask_wise true '''
        #convert to AB mag and select targets
        self.flux_w_ext()
        self.flux_to_mag_ab()
        #target selection, self.lrg is True wherever LRG is
        self.BGS_cuts()
        self.ELG_cuts()
        self.LRG_cuts()
        self.QSO_cuts()
        #sanity checks
        self.assert_same_mask()
        self.assert_TS_not_where_masked()
        
    def assert_same_mask(self):
        '''self.mask should be applied to all self.data.keys() fields'''
        for key in self.data.keys():
            if key.startswith('w1') or key.startswith('w2'):
                assert(np.all(self.data[key].mask == self.mask_wise))
            else:
                assert(np.all(self.data[key].mask == self.mask))
    def assert_TS_not_where_masked(self):
        assert(np.all( self.bgs[self.mask] == False )) #no galaxies should be selected where mask is true
        assert(np.all( self.elg[self.mask] == False )) 
        assert(np.all( self.lrg[self.mask_wise] == False )) 
        assert(np.all( self.qso[self.mask_wise] == False ))
    def count_total(self):
        '''return total number of objects'''
        return self.data['ra'].size
    def count_not_masked(self):
        '''return total number of NOT masked objects'''
        return np.where(self.data['gmag'].mask == False)[0].size

    def flux_w_ext(self):
        for b in self.bands:
            self.data[b+'flux_ext']= self.data[b+'flux']/self.data[b+'_ext']
    def flux_to_mag_ab(self):
        for b in self.bands:
            self.data[b+'mag']= 22.5 -2.5*np.log10(self.data[b+'flux_ext'])
            self.data[b+'mag_ivar']= np.power(np.log(10.)/2.5*self.data[b+'flux'], 2)* self.data[b+'flux_ivar']			
        #non-fundamental
    def update_masks_for_everything(self, mask=None,mask_wise=None):
        if mask is not None: self.mask= np.any((self.mask,mask),axis=0)
        if mask_wise is not None: self.mask_wise= np.any((self.mask_wise,mask_wise),axis=0)  
        #update masks
        for key in self.data.keys():
            if key.startswith('w1') or key.startswith('w2'):
                self.data[key].mask= self.mask_wise #data mask is mask_wise
            else: 
                self.data[key].mask= self.mask
        #redo TS
        self.BGS_cuts()
        self.ELG_cuts()
        self.LRG_cuts()
        self.QSO_cuts()
        #sanity checks
        self.assert_same_mask()
        self.assert_TS_not_where_masked()
	
