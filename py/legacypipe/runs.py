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
        # Brick 1847p145 for sky tests
        I = np.flatnonzero(np.isin(ccds.expnum, [634440, 634064, 431192]) *
                           (ccds.ccdname == 'S16'))
                                   #(ccds.ccdname == 'S17'))
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
    None: LegacySurveyData,
}

def get_survey(name, **kwargs):
    survey_class = runs[name]
    survey = survey_class(**kwargs)
    return survey
