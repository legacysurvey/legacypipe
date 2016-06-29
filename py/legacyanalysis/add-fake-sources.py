from tractor import *

from legacypipe.runbrick import *
from legacypipe.common import *

from astrometry.util.multiproc import *

'''
Test out a code structure for inserting fake galaxies into images, by
overriding the Decals class & DecamImage class.
'''

class FakeDecals(LegacySurveyData):
    def __init__(self, survey_dir=None, fake_sources=[]):
        super(FakeDecals, self).__init__(survey_dir=survey_dir)
        self.fake_sources = fake_sources

    def get_image_object(self, t):
        return FakeImage(self, t)

class FakeImage(DecamImage):
    def __init__(self, survey, t):
        super(FakeImage, self).__init__(survey, t)

    def get_tractor_image(self, **kwargs):
        tim = super(FakeImage, self).get_tractor_image(**kwargs)

        print 'Adding fake sources to', tim

        for src in self.survey.fake_sources:
            patch = src.getModelPatch(tim)
            if patch is None:
                continue
            print 'Adding', src
            patch.addTo(tim.getImage())
        
        return tim



def main():
    survey = FakeDecals()
    # Our list of fake sources to insert into images...
    survey.fake_sources = [PointSource(RaDecPos(0., 0.),
                                       NanoMaggies(g=1000, r=1000, z=1000))]

    # Now, we can either call run_brick()....
    run_brick(None, survey, radec=(0.,0.), width=100, height=100,
              stages=['image_coadds'])

    # ... or the individual stages in the pipeline:
    if False:
        mp = multiproc()
        kwa = dict(W = 100, H=100, ra=0., dec=0., mp=mp, brickname='fakey')
        kwa.update(survey=survey)
        R = stage_tims(**kwa)
        kwa.update(R)
        R = stage_image_coadds(**kwa)
        # ...


if __name__ == '__main__':
    main()
    
