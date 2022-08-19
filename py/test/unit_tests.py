import unittest

class TestIterWrapper(unittest.TestCase):
    def test_peek(self):
        from legacypipe.utils import iterwrapper

        def my_iterator(n):
            for i in range(n):
                print('Yielding', i)
                yield i

        # N = 3
        # it = iterwrapper(my_iterator(N), N)
        # for j in range(N):
        #     print('Iterating:')
        #     x = next(it)
        #     print('Got', x)
        # 
        # print()
        # print('Reset')
        N = 5
        it = iterwrapper(my_iterator(N), N)

        print('Peeking')
        x0 = it.peek()
        print('Got', x0)
        print('Peeking again')
        x1 = it.peek()
        print('Got', x1)
        print('Peeking again')
        x2 = it.peek()
        print('Got', x2)

        print('Popping', x1)
        it.pop(x1)

        xp = 7
        print('Pushing', xp)
        it.push(xp)

        print('Iterating...')
        X = list(it)
        print('Got:', X)
        # for j in range(N):
        #     print('Iterating:')
        #     x = next(it)
        #     print('Got', x)
        self.assertTrue(X == [0, 2, 7, 3, 4])

class TestOneblob(unittest.TestCase):

    def test_modelsel(self):
        from legacypipe.oneblob import _select_model

        nparams = dict(psf=2, rex=3, exp=5, dev=5, comp=9)
        galaxy_margin = 3.**2 + (nparams['exp'] - nparams['psf'])

        chisqs = dict(psf=0, rex=0)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'none')

        chisqs = dict(psf=500, rex=0)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'psf')

        chisqs = dict(psf=0, rex=500)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'rex')

        chisqs = dict(psf=500, rex=501)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'psf')

        chisqs = dict(psf=500, rex=503)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'psf')

        chisqs = dict(psf=500, rex=510)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'rex')

        chisqs = dict(psf=500, rex=504, exp=505)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'psf')

        chisqs = dict(psf=500, rex=505, exp=505)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'rex')

        chisqs = dict(psf=500, rex=505, exp=520)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'exp')

        chisqs = dict(psf=5000, rex=5005, exp=5020)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'psf')

        chisqs = dict(psf=5000, rex=5005, exp=5051)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'exp')

        chisqs = dict(psf=5000, rex=5005, exp=5051, dev=5052)
        mod = _select_model(chisqs, nparams, galaxy_margin)
        self.assertTrue(mod == 'dev')


if __name__ == '__main__':
    unittest.main()
    #t = TestIterWrapper()
    #t.test_peek()
