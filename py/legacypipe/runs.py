from __future__ import print_function

from legacypipe.survey import LegacySurveyData

class DecamSurvey(LegacySurveyData):
    def filter_ccd_kd_files(self, fns):
        return [fn for fn in fns if 'decam' in fn]
    def filter_ccds_files(self, fns):
        return [fn for fn in fns if 'decam' in fn]
    def get_default_release(self):
        return 9002

class NinetyPrimeMosaic(LegacySurveyData):
    def filter_ccd_kd_files(self, fns):
        return [fn for fn in fns if ('90prime' in fn) or ('mosaic' in fn)]
    def filter_ccds_files(self, fns):
        return [fn for fn in fns if ('90prime' in fn) or ('mosaic' in fn)]
    def get_default_release(self):
        return 9003

class M33SurveyData(LegacySurveyData):
    def filter_ccd_kd_files(self, fns):
        return [fn for fn in fns if 'm33' in fn]
    def filter_ccds_files(self, fns):
        return [fn for fn in fns if 'm33' in fn]
    def get_default_release(self):
        return 9002

runs = {
    'decam': DecamSurvey,
    '90prime-mosaic': NinetyPrimeMosaic,
    'south': DecamSurvey,
    'north': NinetyPrimeMosaic,
    'm33': M33SurveyData,
    None: LegacySurveyData,
}

def get_survey(name, **kwargs):
    survey_class = runs[name]
    survey = survey_class(**kwargs)
    return survey
