import unittest

class TestOneblob(unittest.TestCase):

    def test_modelsel(self):
        from legacypipe.oneblob import _select_model

        nparams = dict(ptsrc=2, rex=3, exp=5, dev=5, comp=9)
        galaxy_margin = 3.**2 + (nparams['exp'] - nparams['ptsrc'])
        rex = True

        chisqs = dict(ptsrc=0, rex=0)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'none')

        chisqs = dict(ptsrc=500, rex=0)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'ptsrc')

        chisqs = dict(ptsrc=0, rex=500)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'rex')

        chisqs = dict(ptsrc=500, rex=501)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'ptsrc')

        chisqs = dict(ptsrc=500, rex=503)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'ptsrc')

        chisqs = dict(ptsrc=500, rex=510)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'rex')

        chisqs = dict(ptsrc=500, rex=504, exp=505)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'ptsrc')

        chisqs = dict(ptsrc=500, rex=505, exp=505)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'rex')

        chisqs = dict(ptsrc=500, rex=505, exp=520)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'exp')

        chisqs = dict(ptsrc=5000, rex=5005, exp=5020)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'ptsrc')

        chisqs = dict(ptsrc=5000, rex=5005, exp=5051)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'exp')

        chisqs = dict(ptsrc=5000, rex=5005, exp=5051, dev=5052)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'dev')


if __name__ == '__main__':
    unittest.main()
