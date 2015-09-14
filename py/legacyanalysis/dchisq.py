if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
from astrometry.util.fits import *
from astrometry.util.plotutils import *

if __name__ == '__main__':
    ps = PlotSequence('dchisq')

    for fn in [#'dr1/tractor/240/tractor-2402p062.fits',
        #'dr2b/tractor/240/tractor-2402p062.fits',
        # After fixing dchisq indexing bug
        'tractor-2402p062.fits',
               ]:
        print
        print fn
        T = fits_table(fn)
        print len(T), 'sources'

        T.ispsf = np.array([t.strip() == 'PSF' for t in T.type])
        T.isexp = np.array([t.strip() == 'EXP' for t in T.type])
        T.isdev = np.array([t.strip() == 'DEV' for t in T.type])
        T.iscomp = np.array([t.strip() == 'COMP' for t in T.type])

        P = T[T.ispsf]
        print len(P), 'PSFs'
        E = T[T.isexp]
        print len(E), 'exps'
        D = T[T.isdev]
        print len(D), 'deVs'
        C = T[T.iscomp]
        print len(C), 'comps'

        # Find an example where the type is EXP but EXP is not better than PSF.
        d = (E.dchisq[:,2] - E.dchisq[:,0]) / 2.
        cut = np.maximum(4.5, (E.dchisq[:,0] / 2.) * 0.02)
        I = np.flatnonzero(d < cut)
        for pt,e,c,dchi,bx,by in zip(E.dchisq[I,0], E.dchisq[I,2], cut[I],d[I], E.bx[I], E.by[I]):
            print 'ptsrc:', pt, 'exp', e, 'dchi', dchi, 'cut:', c, 'bx,by', int(bx),int(by)
        print len(I), 'EXP and bad.'
        bad = E[I]
        bad.writeto('bad-exp.fits')
        
        mn,mx = 1e-2, 1e8
        xx = np.logspace(np.log10(mn), np.log10(mx), 100)
        
        # plt.clf()
        # p1 = plt.plot(T.dchisq[:,0], T.dchisq[:,1], 'r.')
        # p2 = plt.plot(T.dchisq[:,0], T.dchisq[:,2], 'b.')
        # p3 = plt.plot(T.dchisq[:,0], T.dchisq[:,3], 'm.')
        # plt.legend([p1[0],p2[0],p3[0]], ['dchisq for deV',
        #                                  'dchisq for exp',
        #                                  'dchisq for comp'],
        #            loc='upper left')
        # plt.xlabel('DCHISQ (PSF)')
        # plt.ylabel('DCHISQ (other)')
        # plt.title('All objects')
        # plt.xscale('log')
        # plt.yscale('log')
        # ax = plt.axis()
        # mn = min(ax[0], ax[2])
        # mx = max(ax[1], ax[3])
        # plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.1)
        # 
        # xx = np.logspace(np.log10(mn), np.log10(mx), 100)
        # plt.plot(xx, xx+4.5, 'b-', alpha=0.1)
        # plt.plot([mn,mx],[mn * 1.02, mx * 1.02], 'r-', alpha=0.1)
        # 
        # plt.axis([mn,mx,mn,mx])
        # ps.savefig()
        
        # for X,name in [(P, 'PSF'), (E, 'exp'), (D, 'deV'), (C, 'comp')]:
        #     plt.clf()
        #     p1 = plt.plot(X.dchisq[:,0], X.dchisq[:,1], 'r.')
        #     p2 = plt.plot(X.dchisq[:,0], X.dchisq[:,2], 'b.')
        #     p3 = plt.plot(X.dchisq[:,0], X.dchisq[:,3], 'm.')
        #     plt.legend([p1[0],p2[0],p3[0]], [
        #         'dchisq for deV', 'dchisq for exp', 'dchisq for comp'],
        #         loc='upper left')
        #     plt.xlabel('DCHISQ (PSF)')
        #     plt.ylabel('DCHISQ (other)')
        #     plt.title('Objects with type = %s' % name)
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     #ax = plt.axis()
        #     #mn = min(ax[0], ax[2])
        #     #mx = max(ax[1], ax[3])
        #     plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.1)
        # 
        #     plt.plot(xx, xx+4.5, 'b-', alpha=0.1)
        #     plt.plot([mn,mx],[mn * 1.02, mx * 1.02], 'r-', alpha=0.1)
        #     
        #     plt.axis([mn,mx,mn,mx])
        #     ps.savefig()
        # 
        # 
        # for iother,oname in [(1, 'deV'), (2, 'exp'), (3, 'comp')]:
        #     plt.clf()
        #     p1 = plt.plot(P.dchisq[:,0], P.dchisq[:,iother], 'g.')
        #     p2 = plt.plot(E.dchisq[:,0], E.dchisq[:,iother], 'r.')
        #     p3 = plt.plot(D.dchisq[:,0], D.dchisq[:,iother], 'b.')
        #     p4 = plt.plot(C.dchisq[:,0], C.dchisq[:,iother], 'm.')
        #     plt.legend([p1[0],p2[0],p3[0],p4[0]],
        #                ['type = PSF', 'type = EXP', 'type = DEV', 'type = COMP'],
        #                loc='upper left')
        #     plt.xlabel('DCHISQ (PSF)')
        #     plt.ylabel('DCHISQ (%s)' % oname)
        #     plt.title('All objects')
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     #ax = plt.axis()
        #     #mn = min(ax[0], ax[2])
        #     #mx = max(ax[1], ax[3])
        #     plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.1)
        # 
        #     plt.plot(xx, xx+4.5, 'b-', alpha=0.1)
        #     plt.plot([mn,mx],[mn * 1.02, mx * 1.02], 'r-', alpha=0.1)
        #     
        #     plt.axis([mn,mx,mn,mx])
        #     ps.savefig()



        for normed in [False, True]:
            plt.clf()
            cc = ['g', 'r', 'b', 'm']
            modnames = ['ptsrc', 'dev', 'exp', 'comp']
            lp = []
            mx = 0
            for i,(name,X) in enumerate(zip(modnames, [P, D, E, C])):
                n,b,p = plt.hist(X.dchisq[:,i], range=(0, 1000), bins=25,
                                 histtype='step', color=cc[i], normed=normed)
                lp.append(p[0])
                mx = max(mx, max(n))
            plt.legend(lp, modnames)
            plt.xlabel('DCHISQ')
            plt.ylim(0, 1.1*mx)
            plt.axvline(25., color='r', alpha=0.2)
            ps.savefig()


        
        plt.clf()
        cc = ['g', 'r', 'b', 'm']
        modnames = ['ptsrc', 'dev', 'exp', 'comp']
        lp = []
        mx = 0
        for i,(name,X) in enumerate(zip(modnames, [P, D, E, C])):
            #n,b,p = plt.hist(X.dchisq[:,i], range=(0, 1000), bins=100, histtype='step', color=cc[i])
            try:
                n,b,p = plt.hist(X.dchisq[:,i], range=(0, 100), bins=50, histtype='step', color=cc[i])
                lp.append(p[0])
                mx = max(mx, max(n))
            except:
                continue
        plt.legend(lp, modnames)
        plt.xlabel('DCHISQ')
        plt.ylim(0, 1.1*mx)
        plt.axvline(25., color='r', alpha=0.2)
        ps.savefig()



        plt.clf()
        cc = ['g', 'r', 'b', 'm']
        modnames = ['ptsrc', 'dev', 'exp', 'comp']
        lp = []
        mx = 0
        for i,(name,X) in enumerate(zip(modnames, [P, D, E, C])):
            #n,b,p = plt.hist(X.dchisq[:,i], range=(0, 1000), bins=100, histtype='step', color=cc[i])
            try:
                n,b,p = plt.hist(X.dchisq[:,0], range=(0, 100), bins=50, histtype='step', color=cc[i])
                lp.append(p[0])
                mx = max(mx, max(n))
            except:
                continue

            print
            #print np.count_nonzero(X.dchisq[:,0] > 1000), name, 'have DCHISQ[PSF] > 1000', 'Of', len(X), 'total'
            print np.count_nonzero(X.dchisq[:,0] > 50), name, 'have DCHISQ[PSF] > 50', 'Of', len(X), 'total'
            print np.count_nonzero(X.dchisq[:,0] < 50), name, 'have DCHISQ[PSF] < 50', 'Of', len(X), 'total'
            print np.count_nonzero(X.dchisq[:,0] < 25), name, 'have DCHISQ[PSF] < 25', 'Of', len(X), 'total'
            print np.count_nonzero(X.dchisq[:,i] < 25), name, 'have DCHISQ[%s] < 25' % name, 'Of', len(X), 'total'
            print np.count_nonzero(np.max(X.dchisq, axis=1) < 25), name, 'have max(DCHISQ) < 25, Of', len(X), 'total'

            I = np.flatnonzero(np.max(X.dchisq, axis=1) < 25)
            X[I].writeto('bad-%s.fits' % name)
            
        plt.legend(lp, modnames)
        plt.xlabel('DCHISQ (PSF)')
        plt.ylim(0, 1.1*mx)
        plt.axvline(25., color='r', alpha=0.2)
        ps.savefig()

        img = plt.imread('decals-2402p062-image.jpg')
        img = np.flipud(img)
        print 'Image', img.shape, img.dtype
        H,W,three = img.shape

        img = plt.imread('decals-2402p062-image.jpg')
        img = np.flipud(img)
        mod = plt.imread('decals-2402p062-model.jpg')
        mod = np.flipud(mod)
        
        C = T[T.dchisq[:,0] < 60]
        print len(C), 'sources have dchisq(psf) < 60'
        
        C.cut(np.lexsort((C.dchisq[:,0], C.type)))

        C.cut(np.max(C.decam_flux, axis=1) > 0.)
        print len(C), 'sources have at least one positive flux'
        
        detsns = [fitsio.read('detsn-2402p062-%s.fits' % b)
                   for b in 'zrg']


        # Select some marginal classifications.
        d = T.dchisq[:, 2] - T.dchisq[:,0]
        I = np.flatnonzero((d > 7) * (d < 11) *
                           np.logical_or(T.ispsf, T.isexp))
        print len(I), 'marginal ptsrc/exp classifications'

        # plt.clf()
        # ha = dict(range=(0,25), bins=25, histtype='step')
        # plt.hist(d, color='k', **ha)
        # plt.hist(d[T.ispsf], color='g', **ha)
        # plt.hist(d[T.isexp], color='r', **ha)
        # ps.savefig()
        

        I = np.flatnonzero(np.logical_or(T.ispsf * (d > 7) * (d < 9),
                                         T.isexp * (d < 11) * (d > 9)))
        print len(I), 'marginal ptsrc/exp classifications based on chisq alone'

        C = T[I]
        C.d = d[I]
        #C.cut(np.lexsort((C.dchisq[:,0], C.type)))
        #C.cut(np.lexsort((C.d, C.type)))
        C.cut(np.argsort(-C.d))
        
        for page in range(2):
            plt.clf()
            #rows,cols = 7,10
            rows,cols = 9,13
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0, wspace=0)
            xytext = []
            i0 = page * rows * cols
            for i in range(rows*cols):
                if i0 + i >= len(C):
                    break
                plt.subplot(rows, cols, i+1)
                S = 20
                c = C[i0 + i]
                y0,y1 = max(0, c.by-S), min(H, c.by+S+1)
                x0,x1 = max(0, c.bx-S), min(W, c.bx+S+1)
                plt.imshow(img[y0:y1, x0:x1], interpolation='nearest', origin='lower')
                txt = '%s (%.0f)' % (c.type, c.d)
                plt.text(0, 0, txt, ha='left', va='bottom', color='red', fontsize=8)
                plt.xticks([])
                plt.yticks([])
    
                xytext.append((x0,x1,y0,y1, txt))
    
            plt.suptitle('Sources with marginal classification')
            ps.savefig()
        #sys.exit(0)


        
        
        #for page in range(5):
        if False:
            plt.clf()
            rows,cols = 7,10
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0, wspace=0)

            xytext = []
            
            i0 = 0
            for i in range(rows*cols):
                plt.subplot(rows, cols, i+1)
                S = 20
                if page >= 1 and i%cols == 0:
                    mindchi = 27 + i/cols + (page-1)*rows
                    print 'Setting min dchisq', mindchi
                    C.cut(np.array([t.strip() == 'PSF' for t in C.type]) * (C.dchisq[:,0] > mindchi))
                    i0 = -i
                c = C[i + i0]
                y0,y1 = max(0, c.by-S), min(H, c.by+S+1)
                x0,x1 = max(0, c.bx-S), min(W, c.bx+S+1)
                plt.imshow(img[y0:y1, x0:x1], interpolation='nearest', origin='lower')
                txt = '%s %.0f' % (c.type, c.dchisq[0])
                plt.text(0, 0, txt, ha='left', va='bottom', color='red', fontsize=8)
                plt.xticks([])
                plt.yticks([])

                xytext.append((x0,x1,y0,y1, txt))

            plt.suptitle('Sources with dchisq(psf) < 50')
            ps.savefig()

            for i,(x0,x1,y0,y1,txt) in enumerate(xytext):
                plt.subplot(rows, cols, i+1)
                plt.imshow(mod[y0:y1, x0:x1], interpolation='nearest', origin='lower')
                plt.text(0, 0, txt, ha='left', va='bottom', color='red', fontsize=8)
                plt.xticks([])
                plt.yticks([])
            ps.savefig()
            
            for i,(x0,x1,y0,y1,txt) in enumerate(xytext):
                plt.subplot(rows, cols, i+1)

                sn = np.dstack((detsn[y0:y1,x0:x1] for detsn in detsns))
                lo,hi = -2, 8
                sn = np.clip((sn - lo) / (hi - lo), 0., 1.)
                plt.imshow(sn, interpolation='nearest', origin='lower')
                plt.text(0, 0, txt, ha='left', va='bottom', color='red', fontsize=8)
                plt.xticks([])
                plt.yticks([])
            ps.savefig()
                
            
        #continue


        mn,mx = 1e-2, 1e8
        xx = np.logspace(np.log10(mn), np.log10(mx), 100)

        for iother,X,oname in [(1, D, 'deV'), (2, E, 'exp'), (3, C, 'comp')]:
            plt.clf()
            p1 = plt.plot(X.dchisq[:,0], X.dchisq[:,iother], 'k.')
            plt.legend([p1[0]],
                       ['type = %s' % oname],
                       loc='upper left')
            plt.xlabel('DCHISQ (PSF)')
            plt.ylabel('DCHISQ (%s)' % oname)
            plt.title('All objects')
            plt.xscale('log')
            plt.yscale('log')

            plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.1)
            plt.plot(xx, xx+4.5, 'b-', alpha=0.1)
            plt.plot([mn,mx],[mn * 1.02, mx * 1.02], 'r-', alpha=0.1)
            
            plt.axis([mn,mx,mn,mx])
            ps.savefig()


        for iother,X,oname in [(1, D, 'deV'), (2, E, 'exp'), (3, C, 'comp')]:
            plt.clf()

            lo,hi = 1e-1, 1e3

            p1 = plt.plot(X.dchisq[:,0], np.clip(X.dchisq[:,iother] / X.dchisq[:,0], lo, hi), 'k.')
            plt.legend([p1[0]], ['type = %s' % oname], loc='upper left')
            plt.xlabel('DCHISQ (PSF)')
            plt.ylabel('DCHISQ (%s) / DCHISQ (PSF)' % oname)
            plt.title('All objects')
            plt.xscale('log')
            plt.yscale('log')
            plt.axhline(1., color='k', alpha=0.1)
            plt.plot(xx, (xx+9) / xx, 'b-', alpha=0.1)
            plt.axhline(1.01, color='r', alpha=0.1)
            
            plt.axis([mn,mx,lo,hi])
            ps.savefig()
            
            #lo,hi = 0.5, 10.
            lo,hi = 0.9, 2.
            plt.axis([mn,mx,lo,hi])
            ps.savefig()

