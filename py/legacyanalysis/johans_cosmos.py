import matplotlib
matplotlib.use('Agg') #display backend
import matplotlib.pyplot as plt
import numpy as np
import os

from legacyanalysis.pathnames import get_indir
indir= get_indir('cosmos')

#CREATES THE CATALOG LIST
catList = np.array(["catalog-R3-R4.fits",
"catalog-R2-R4.fits",
"catalog-R2-R3.fits",
"catalog-R1-R4.fits",
"catalog-R1-R3.fits",
"catalog-R1-R2.fits"])
for cnt,cat in enumerate(catList): catList[cnt]= os.path.join(indir,cat) 
# EACH CATALOG NEEDS TO HAVE THE TYPICAL DECALS CATALOG ENTRIES WITH "_1" AND "_2" APPENDED FOR DR2 and DR3

# DEFINES THE GAUSSIAN FUNCTION
gfun = lambda x, m0, s0 : st.norm.pdf(x,loc=m0,scale=s0)
#OPENS A FILE TO WRITE OUTPUTS
f=open(os.path.join(get_outdir('cosmos'),"depth-comparisonp.txt"),"w")
f.write("20<g<21.5 \n")
# CREATES A FIGURE
plt.figure(2,(5,5))
plt.axes([0.17,0.15,0.75,0.75])
# PLOT THE EXPECTED NORMAL DISTRIBUTION
plt.plot(np.arange(-10,6,0.1), st.norm.pdf(np.arange(-10,6,0.1),loc=0,scale=1), 'k--', lw=2, label='N(0,1)')

# LOOPS OVER MATCHED CATALOGS
for ii, el in enumerate(catList):
    hdu=fits.open(el)
    dr2=hdu[1].data
        # DEFINES MAGNITUDES TO SELECT A MAGNITUDE BIN
    g_mag_dr2 = 22.5 - 2.5 * np.log10(dr2['decam_flux_2'].T[1] / dr2['decam_mw_transmission_2'].T[1])
    r_mag_dr2 = 22.5 - 2.5 * np.log10(dr2['decam_flux_2'].T[2] / dr2['decam_mw_transmission_2'].T[2])
    z_mag_dr2 = 22.5 - 2.5 * np.log10(dr2['decam_flux_2'].T[4] / dr2['decam_mw_transmission_2'].T[4])
        # SELECT A POPULATION OF SOURCES
    sel = (dr2['type_2'] == "PSF")&(g_mag_dr2>20)&(g_mag_dr2<21.5)
        # COMPARES THE PHOTOMETRIC OUTPUTS 
    df_g = dr2[sel]['decam_flux_1'].T[1] / dr2[sel]['decam_mw_transmission_1'].T[1] - dr2[sel]['decam_flux_2'].T[1] / dr2[sel]['decam_mw_transmission_2'].T[1] 
    sigma_g = (1./dr2[sel]['decam_flux_ivar_1'].T[1] + 1./dr2[sel]['decam_flux_ivar_2'].T[1])**(-0.5)
        # CREATES THE HISTOGRAM 
    nnn,bbb, ppp=plt.hist(df_g * sigma_g, bins=np.arange(-4,4.5,0.25), weights = np.ones_like(df_g)/area, histtype='step', label=str(ii), normed=True)
        #FITS A GAUSSIAN TO THE HISTOGRAM AND WRITES THE OUTPUT
    out = cu(gfun,(bbb[1:]+bbb[:-1])/2.,nnn,p0=(0,1))
    f.write(el + '\n')
    f.write(str(ii) + " " + str(out[0]))
    f.write('\n')
    
plt.xlabel('(g(Ri)-g(FD))/sqrt(var_g(Ri) + var_g(FD))')
plt.ylabel('Normed counts')
plt.xlim((-4,4))
plt.ylim((0,0.7))
gp = plt.legend(loc=2, fontsize=10)
gp.set_frame_on(False)
pt.title('20<g<21.5 type PSF')
plt.grid()
#SAVES THE PLOT AND CLOSES THE FILE WHERE THINGS WERE WRITTEnp.
plt.savefig(os.path.join(get_outdir('cosmos'),"plotsRc", "comparison-depth-normed-g-20-215-cosmos.png"))
plt.clf()
f.close()
