from __future__ import print_function
import numpy as np

from legacypipe.survey import LegacySurveyData

# class LegacySurveyRun(object):
#     def __init__(self, opt):
#         self.opt = opt
# 
#     def ccds_for_fitting(survey, brick, ccds):
#         print('ccds_for_fitting')
#         return None
#     def __str__(self):
#         return 'Run: ' + str(type(self))
# 
# class Dr3Decals(LegacySurveyRun):
#     def ccds_for_fitting(survey, brick, ccds):
#         return np.flatnonzero(ccds.camera == 'decam')
# 
# class Dr4MzLS(LegacySurveyRun):
#     def ccds_for_fitting(survey, brick, ccds):
#         return np.flatnonzero(np.logical_or(ccds.camera == 'mosaic',
#                                             ccds.camera == '90prime'))
#     #def __str__(self):
#     #    return 'Dr4(MzLS)'
# 
# runs = {
#     'dr3': Dr3Decals,
#     'dr4': Dr4MzLS,
#     None: LegacySurveyRun,
# }
# 
# def get_run(name):
#     return runs[name]

class Dr3Survey(LegacySurveyData):
    pass

runs = {
    'dr3': Dr3Survey,
    None: LegacySurveyData,
}

def get_survey(name, **kwargs):
    survey_class = runs[name]
    survey = survey_class(**kwargs)
    return survey
