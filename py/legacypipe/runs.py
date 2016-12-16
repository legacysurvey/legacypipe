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

class Dr4Survey(LegacySurveyData):
    def ccds_for_fitting(survey, brick, ccds):
        return np.flatnonzero(np.logical_or(ccds.camera == 'mosaic',
                                            ccds.camera == '90prime'))

class Dr4(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-dr4-90prime.fits.gz' in fn or
                  'survey-ccds-dr4-mzlsv2.fits.gz' in fn)]

class Dr4Mzlsv2(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-dr4-mzlsv2.fits.gz' in fn)]


class Dr4Bootes(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-90prime-bootes.fits.gz' in fn or
                  'survey-ccds-mzlsv2thruMarch19.fits.gz' in fn)]

class Dr4Bootes90Prime(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-90prime-bootes.fits.gz' in fn)]

class Dr4BootesMzls(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-mzlsv2thruMarch19.fits.gz' in fn)]

runs = {
    'dr3': Dr3DecalsSurvey,
    'dr4': Dr4,
    'dr4-mzlsv2': Dr4Mzlsv2,
    'dr4-bootes': Dr4Bootes,
    'bootes-90prime': Dr4Bootes90Prime,
    'bootes-mzlsv2thruMarch19': Dr4BootesMzls,
    None: LegacySurveyData,
}

def get_survey(name, **kwargs):
    survey_class = runs[name]
    survey = survey_class(**kwargs)
    return survey
