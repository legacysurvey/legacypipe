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
    #    return [fn for fn in fns if ('90prime' in fn) or ('mosaic' in fn)]
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
    'rerun-ccds': RerunWithCcds,
    None: LegacySurveyData,
}

def get_survey(name, **kwargs):
    survey_class = runs[name]
    survey = survey_class(**kwargs)
    return survey
