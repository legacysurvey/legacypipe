
import pickle
#tr = pickle.load(open('bad.pickle','rb'))
tr = pickle.load(open('bad2.pickle','rb'))
tr.model_kwargs = {}
print('Got', tr)
opt = tr.optimizer
from astrometry.util.plotutils import PlotSequence
opt.ps = PlotSequence('bad2')
opt.getSingleImageUpdateDirections(tr, shared_params=False)
