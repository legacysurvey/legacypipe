from __future__ import print_function
import numpy as np

from legacypipe.survey import LegacySurveyData

class Dr3DecalsSurvey(LegacySurveyData):
    # Do we want/need this cut?
    # def filter_ccds_files(self, fns):
    #     return [fn for fn in fns if
    #             ('survey-ccds-decals.fits.gz' in fn or
    #              'survey-ccds-nondecals.fits.gz' in fn or
    #              'survey-ccds-extra.fits.gz' in fn)]

    def ccds_for_fitting(self, brick, ccds):
        return np.flatnonzero(ccds.camera == 'decam')

class Dr4MzlsSurvey(LegacySurveyData):
    def ccds_for_fitting(survey, brick, ccds):
        return np.flatnonzero(np.logical_or(ccds.camera == 'mosaic',
                                            ccds.camera == '90prime'))
    
class Dr4Bootes(Dr4MzlsSurvey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-90prime.fits.gz' in fn or
                  'survey-ccds-mzls-v2thruMarch19.fits.gz' in fn)]

class MzlsV2(LegacySurveyData):
    def ccds_for_fitting(survey, brick, ccds):
        return np.flatnonzero(ccds.camera == 'mosaic')
    
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-mzls-v2.fits.gz' in fn)]

class MzlsV3(LegacySurveyData):
    def ccds_for_fitting(survey, brick, ccds):
        return np.flatnonzero(ccds.camera == 'mosaic')
    
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-mzls-v3.fits.gz' in fn)]



runs = {
    'dr3': Dr3DecalsSurvey,
    'dr4': Dr4MzlsSurvey,
    'dr4-bootes': Dr4Bootes,
    'mzls-v2': MzlsV2,
    'mzls-v3': MzlsV3,
    None: LegacySurveyData,
}

def get_survey(name, **kwargs):
    survey_class = runs[name]
    survey = survey_class(**kwargs)
    return survey
