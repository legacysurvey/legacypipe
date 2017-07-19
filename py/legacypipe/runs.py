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

    def filter_ccd_kd_files(self, fns):
        return []

class Dr4Survey(LegacySurveyData):
    def ccds_for_fitting(survey, brick, ccds):
        return np.flatnonzero(np.logical_or(ccds.camera == 'mosaic',
                                            ccds.camera == '90prime'))

class Dr4v2(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-dr4-90prime.fits.gz' in fn or
                  'survey-ccds-dr4-mzlsv2.fits.gz' in fn)]

class Dr4v3(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-dr4-90prime.fits.gz' in fn or
                  'survey-ccds-dr4-mzlsv3.fits.gz' in fn)]


class Thirdpixv2(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-thirdpix-v2.fits.gz' in fn)]
                 #('survey-ccds-thirdpix-len2-v2.fits.gz' in fn)]
                 #('survey-ccds-dr4-mzlsv2.fits.gz' in fn)]

class Thirdpixv3(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-thirdpix-v3.fits.gz' in fn)]
                 #('survey-ccds-thirdpix-len2-v3.fits.gz' in fn)]

class Dr4Mzlsv2(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-dr4-mzlsv2.fits.gz' in fn)]

class Dr4Mzlsv3(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-dr4-mzlsv3.fits.gz' in fn)]

class Dr490prime(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-dr4-90prime.fits.gz' in fn)]

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

class Dr4Plus(Dr4Survey):
    def filter_ccds_files(self, fns):
         return [fn for fn in fns if
                 ('survey-ccds-dr4-90prime.fits.gz' in fn or
                  'survey-ccds-dr4-mzlsv2.fits.gz' in fn or
                  'survey-ccds-mzls-runs-16-to-21a.fits.gz' in fn)]

runs = {
    'dr3': Dr3DecalsSurvey,
    'thirdpix-v2': Thirdpixv2,
    'thirdpix-v3': Thirdpixv3,
    'dr4v2': Dr4v2,
    'dr4v3': Dr4v3,
    'dr490prime': Dr490prime,
    'dr4-mzlsv2': Dr4Mzlsv2,
    'dr4-mzlsv3': Dr4Mzlsv3,
    'dr4-bootes': Dr4Bootes,
    'bootes-90prime': Dr4Bootes90Prime,
    'bootes-mzlsv2thruMarch19': Dr4BootesMzls,
    'dr4+': Dr4Plus,
    None: LegacySurveyData,
}

def get_survey(name, **kwargs):
    survey_class = runs[name]
    survey = survey_class(**kwargs)
    return survey
