from legacypipe.survey import LegacySurveyData

class DecamSurvey(LegacySurveyData):
    def filter_ccd_kd_files(self, fns):
        return [fn for fn in fns if 'decam' in fn]
    def filter_ccds_files(self, fns):
        return [fn for fn in fns if 'decam' in fn]
    def filter_annotated_ccds_files(self, fns):
        return [fn for fn in fns if 'decam' in fn]
    def get_default_release(self):
        return 9008

class NinetyPrimeMosaic(LegacySurveyData):
    def filter_ccd_kd_files(self, fns):
        return [fn for fn in fns if ('90prime' in fn) or ('mosaic' in fn)]
    def filter_ccds_files(self, fns):
        return [fn for fn in fns if ('90prime' in fn) or ('mosaic' in fn)]
    def filter_annotated_ccds_files(self, fns):
        return [fn for fn in fns if ('90prime' in fn) or ('mosaic' in fn)]
    def get_default_release(self):
        return 9009

class M33SurveyData(DecamSurvey):
    def ccds_for_fitting(self, brick, ccds):
        import numpy as np
        from astrometry.libkd.spherematch import match_radec
        I, _, _ = match_radec(ccds.ra, ccds.dec, np.array(23.462121), np.array(30.659925), 0.55, nearest=True)
        #I = np.delete(I, np.where((ccds.filter[I] == 'g') * (ccds.expnum[I] != 661055))[0])
        #I = np.delete(I, np.where((ccds.filter[I] == 'z') * (ccds.expnum[I] != 790242))[0])
        return I

class OdinData(LegacySurveyData):
    #def filter_ccd_kd_files(self, fns):
    #    return [fn for fn in fns if ('odin' in fn)]
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('OdinData: kwargs', kwargs)
        self.update_maskbits_bands(['N419', 'N501', 'N673'])
        print('Maskbits:', self.get_maskbits())
        print('Maskbits descriptions:', self.get_maskbits_descriptions())

    def filter_ccds_files(self, fns):
        return [fn for fn in fns if ('odin' in fn)]
    def filter_annotated_ccds_files(self, fns):
        return [fn for fn in fns if ('odin' in fn)]
    def get_default_release(self):
        return 200

class HscData(LegacySurveyData):
    #def filter_ccd_kd_files(self, fns):
    #    return [fn for fn in fns if ('90prime' in fn) or ('mosaic' in fn)]
    def filter_ccds_files(self, fns):
        return [fn for fn in fns if ('hsc' in fn)]
    def filter_annotated_ccds_files(self, fns):
        return [fn for fn in fns if ('hsc' in fn)]
    def get_default_release(self):
        return 200

class HscCorrData(HscData):
    def filter_ccds(self, ccds):
        import numpy as np
        I = np.flatnonzero([('CORR' in fn) for fn in ccds.image_filename])
        print('HscCorrData: cutting to', len(I), 'of', len(ccds), 'CCDs with CORR in the filename')
        return ccds[I]

class HscCalexpData(HscData):
    def filter_ccds(self, ccds):
        import numpy as np
        I = np.flatnonzero([('CALEXP' in fn) for fn in ccds.image_filename])
        print('HscCalexpData: cutting to', len(I), 'of', len(ccds), 'CCDs with CALEXP in the filename')
        return ccds[I]

class SuprimeData(LegacySurveyData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_maskbits_bands(['I-A-L427',
                                    'I-A-L464',
                                    'I-A-L484',
                                    'I-A-L505',
                                    'I-A-L527',])
        print('Maskbits:', self.get_maskbits())
        print('Maskbits descriptions:', self.get_maskbits_descriptions())

class IbisData(LegacySurveyData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_maskbits_bands(['M411', 'M438', 'M464', 'M490', 'M517'])
        print('Maskbits:', self.get_maskbits())
        print('Maskbits descriptions:', self.get_maskbits_descriptions())

class IbisWideData(IbisData):
    def ccds_for_fitting(self, brick, ccds):
        import numpy as np
        I = np.flatnonzero(['_wide' in o for o in ccds.object])
        print('IBIS-wide run; cutting to', len(I), 'of', len(ccds), 'CCDs with "_wide" in the OBJECT name')
        return I

class IbisSpecialData(IbisData):
    def ccds_for_fitting(self, brick, ccds):
        import numpy as np
        I = np.flatnonzero([('Pyxis' in o or 'NGC7492' in o) for o in ccds.object])
        print('IBIS-wide run; cutting to', len(I), 'of', len(ccds), 'CCDs with matching OBJECT name')
        return I

class ClaudsTestData1(LegacySurveyData):
    def filter_ccds(self, ccds):
        import numpy as np
        I = np.flatnonzero(np.isin(ccds.expnum, [1796492, 1796503, 1796353, 1795584]))
        print('CLAUDS test #1: cutting CCDs to %i of %i on EXPNUM' % (len(I), len(ccds)))
        return ccds[I]

class ClaudsTestData2(LegacySurveyData):
    def filter_ccds(self, ccds):
        import numpy as np
        I = np.flatnonzero(np.isin(ccds.expnum, [1796348, 1796344]))
        print('CLAUDS test #2: cutting CCDs to %i of %i on EXPNUM' % (len(I), len(ccds)))
        return ccds[I]

class ClaudsTestData3(LegacySurveyData):
    # Actually a CFIS test, but just keeping with the naming scheme...
    def filter_ccd_kd_files(self, fns):
        return [fn for fn in fns if ('cfis-cosmos-u' in fn)]

class ClaudsTestData4(LegacySurveyData):
    # Actually a CFIS test, but just keeping with the naming scheme...
    def filter_ccd_kd_files(self, fns):
        return [fn for fn in fns if ('cfis-cosmos-u' in fn)]
    def filter_ccds(self, ccds):
        import numpy as np
        I = np.flatnonzero(np.isin(ccds.expnum, [2571171, 2602665]))
        print('CLAUDS test #4: cutting CCDs to %i of %i on EXPNUM' % (len(I), len(ccds)))
        return ccds[I]

class Dr11Test1(LegacySurveyData):
    def filter_ccds(self, ccds):
        import numpy as np
        # Brick 0301m040 for i-band forced phot tests
        I = np.flatnonzero(np.isin(ccds.expnum, [247958, 400372, 388167, 390912]) *
                           (ccds.ccdname == 'S31'))
        print('DR11 test #1: cutting CCDs to %i of %i on EXPNUM & CCDNAME' % (len(I), len(ccds)))

class Dr11Test2(LegacySurveyData):
    def filter_ccds(self, ccds):
        import numpy as np
        # Brick 1847p145 for sky tests - big galaxy
        #I = np.flatnonzero(np.isin(ccds.expnum, [634440, 634064, 431192]) *
        #                   (ccds.ccdname == 'S16'))

        # Brick 0059m717 - NGC104 globular cluster covers 100% of some chips
        I = np.flatnonzero(np.isin(ccds.expnum, [380296])) # *
        #(ccds.ccdname == 'S16'))

        print('DR11 test #2: cutting CCDs to %i of %i on EXPNUM & CCDNAME' % (len(I), len(ccds)))
        return ccds[I]

class RerunWithCcds(LegacySurveyData):
    def get_brick_by_name(self, brickname):
        # BRUTAL HACK -- runbrick.py's stage_tims first calls
        # get_brick_by_name, then ccds_touching_wcs... save the
        # brickname here for later use when reading the CCDs file!
        self.thebrick = brickname
        return super().get_brick_by_name(brickname)
    def get_ccds(self, **kwargs):
        from astrometry.util.fits import fits_table
        fn = self.find_file('ccds-table', brick=self.thebrick)
        T = fits_table(fn, **kwargs)
        T = self.cleanup_ccds_table(T)
        print('Read', len(T), 'CCDs from', fn)
        return T
    def get_ccd_kdtrees(self):
        return []

class UnionsRun(LegacySurveyData):
    def get_colorschemes(self):
        return ['ugr', 'riz', 'ugriz']
    def get_colorscheme_tag(self, colorscheme):
        # Returns filename tag for this RGB color scheme
        return '-'+colorscheme
    def get_rgb(self, imgs, bands, coadd_bw=None, colorscheme=None, **kwargs):
        from legacypipe.survey import rgb_stretch_factor, sdss_rgb
        print('get_rgb: colorscheme', colorscheme, 'bands', bands, 'N imgs:', len(imgs))

        # for coadd_bw (code from survey.py)
        kwa = {}
        bw = self.coadd_bw if coadd_bw is None else coadd_bw
        if bw and len(bands) == 1:
            rgb = rgb.sum(axis=2)
            kwa = dict(cmap='gray')

        if colorscheme == 'ugriz':
            rgbscales=dict(u = (0, 1.8 *  rgb_stretch_factor),
                           g = (0, 15.0 * rgb_stretch_factor),
                           r = (0, 6.0 *  rgb_stretch_factor),
                           i = (0, 4.0 *  rgb_stretch_factor),
                           z = (0, 4.0 *  rgb_stretch_factor),
                          )
            #rgbvec = dict(
            #    u = (0.0, 0.0, 0.6),
            #    g = (0.0, 0.2, 0.4),
            #    r = (0.0, 0.6, 0.0),
            #    i = (0.4, 0.2, 0.0),
            #    z = (0.6, 0.0, 0.0))

            # downweight u
            rgbvec = dict(
                u = (0.0 , 0.0 , 0.4),
                g = (0.0 , 0.05, 0.6),
                r = (0.0 , 0.65, 0.0),
                i = (0.35, 0.3 , 0.0),
                z = (0.65, 0.0 , 0.0))

            I = 0
            for img,band in zip(imgs, bands):
                _,scale = rgbscales[band]
                img = np.maximum(0, img * scale + m)
                I = I + img
            I /= len(bands)
            if Q is not None:
                fI = np.arcsinh(Q * I) / np.sqrt(Q)
                I += (I == 0.) * 1e-6
                I = fI / I
            H,W = I.shape
            rgb = np.zeros((H,W,3), np.float32)
            for img,band in zip(imgs, bands):
                _,scale = rgbscales[band]
                rf,gf,bf = rgbvec[band]
                if mnmx is None:
                    v = (img * scale + m) * I
                else:
                    mn,mx = mnmx
                    v = ((img * scale + m) - mn) / (mx - mn)
                if clip:
                    v = np.clip(v, 0, 1)
                if rf != 0.:
                    rgb[:,:,0] += rf*v
                if gf != 0.:
                    rgb[:,:,1] += gf*v
                if bf != 0.:
                    rgb[:,:,2] += bf*v
            return rgb,{}

        elif colorscheme in ['ugr', 'riz']:
            ii = [i for i,b in zip(imgs,bands) if b in colorscheme]
            bb = [b for i,b in zip(imgs,bands) if b in colorscheme]
            if colorscheme == 'ugr':
                scales = dict(
                    u =    (2, 1.5 * rgb_stretch_factor),
                    g =    (1, 6.0 * rgb_stretch_factor),
                    r =    (0, 3.4 * rgb_stretch_factor),
                )
            elif colorscheme == 'riz':
                scales = dict(
                    r =    (2, 5.0 * rgb_stretch_factor),
                    i =    (1, 3.0 * rgb_stretch_factor),
                    z =    (0, 4.0 * rgb_stretch_factor),
                )
            rgb = sdss_rgb(ii, bb, scales=scales)
            if bw and len(bands) == 1:
                rgb = rgb.sum(axis=2)
            return rgb,kwa
        return super().get_rgb(img, bands, coadd_bw=coadd_bw, **kwargs)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_maskbits_bands(['u', 'g', 'r', 'i', 'z'])

runs = {
    'decam': DecamSurvey,
    '90prime-mosaic': NinetyPrimeMosaic,
    'south': DecamSurvey,
    'north': NinetyPrimeMosaic,
    'm33': M33SurveyData,
    'odin': OdinData,
    'hsc': HscData,
    'hsc-corr': HscCorrData,
    'hsc-calexp': HscCalexpData,
    'rerun-ccds': RerunWithCcds,
    'suprime': SuprimeData,
    'ibis': IbisData,
    'ibis-wide': IbisWideData,
    'ibis-special': IbisSpecialData,
    'clauds-test-1': ClaudsTestData1,
    'clauds-test-2': ClaudsTestData2,
    'clauds-test-3': ClaudsTestData3,
    'clauds-test-4': ClaudsTestData4,
    'dr11-test-1': Dr11Test1,
    'dr11-test-2': Dr11Test2,
    'unions': UnionsRun,
    None: LegacySurveyData,
}

def get_survey(name, **kwargs):
    survey_class = runs[name]
    survey = survey_class(**kwargs)
    return survey
