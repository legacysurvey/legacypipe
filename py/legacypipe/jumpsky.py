import numpy as np
from tractor.splinesky import SplineSky
'''
This class implements a version of SplineSky where there is a discrete jump
in sky level at the center of the chip.  It was created to handle DECam chip S30,
where the two amps have different gains, resulting in a break in the sky level.
'''
from tractor.splinesky import median_estimator
class JumpSky(SplineSky):

    @staticmethod
    def BlantonMethod(image, mask, gridsize, xbreak, estimator=median_estimator, **kwargs):
        #box = gridsize // 2
        box = 100
        # Compute medians in a box width (and full height) on
        # either side of the break, and use the difference between
        # them as the jump.
        msk = None
        img = image[:, xbreak-box:xbreak]
        if mask is not None:
            msk = mask[:, xbreak-box:xbreak]
        mleft = estimator(img, msk, **kwargs)
        assert(np.isfinite(mleft))
        img = image[:, xbreak:xbreak+box]
        if mask is not None:
            msk = mask[:, xbreak:xbreak+box]
        mright = estimator(img, msk, **kwargs)
        assert(np.isfinite(mright))
        jump = mright - mleft

        # Apply offset to image and produce regular SplineSky model.
        img = np.empty_like(image)
        img[:, :xbreak] = image[:, :xbreak]
        img[:, xbreak:] = image[:, xbreak:] - jump
        ss = SplineSky.BlantonMethod(img, mask, gridsize, estimator=estimator, **kwargs)
        xgrid = ss.xgrid
        ygrid = ss.ygrid
        gridvals = ss.spl(ss.xgrid, ss.ygrid).T
        return JumpSky(xgrid, ygrid, gridvals, xbreak=xbreak, xjump=jump)

    def __init__(self, xgrid, ygrid, gridvals, xbreak=None, xjump=None, **kwargs):
        super().__init__(xgrid, ygrid, gridvals, **kwargs)
        self.xbreak = xbreak
        self.xjump = xjump

    def copy(self):
        c = super().copy()
        print('Copy:', type(c))
        c.xbreak = self.xbreak
        c.xjump = self.xjump
        return c

    def scale(self, s):
        super().scale(s)
        self.xjump *= s

    def evaluateGrid(self, xvals, yvals):
        g = super().evaluateGrid(xvals, yvals)
        I = np.flatnonzero(xvals + self.x0 >= self.xbreak)
        if len(I):
            g[:,I] += self.xjump
        return g

    def to_fits_table(self):
        T = super().to_fits_table()
        T.xbreak = np.atleast_1d(self.xbreak).astype(np.int32)
        T.xjump = np.atleast_1d(self.xjump).astype(np.float32)
        return T

    @classmethod
    def from_fits_row(cls, t):
        sky = cls(t.xgrid, t.ygrid, t.gridvals, t.xbreak, t.xjump, order=t.order)
        sky.shift(t.x0, t.y0)
        return sky
