from __future__ import print_function
import os
import numpy as np

from legacypipe.forced_photom_decam import main as forced_photom_main
from legacypipe.forced_photom_decam import get_parser as forced_photom_parser
from legacypipe.runcosmos import CosmosDecals
    
def main():
    parser = forced_photom_parser()
    parser.add_argument('--subset', type=int, help='COSMOS subset number [0 to 4]', default=0)
    opt = parser.parse_args()
    survey = CosmosDecals(subset=opt.subset)
    
    np.random.seed(1000000 + opt.subset)
    
    return forced_photom_main(survey=survey, opt=opt)
    
if __name__ == '__main__':
    import sys
    sys.exit(main())
