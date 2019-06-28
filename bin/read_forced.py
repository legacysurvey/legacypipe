from astropy.io import fits
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
basedir='/project/projectdirs/uLens/ZTF/Tractor/data/ZTF18abcfdzu/tractor/tractor/cus'

cat=glob.glob(basedir+'/tractor-custom*.fits')[0]
allforced=glob.glob(basedir+'/forced*.fits')
print(len(allforced),'datapoints')
'''
with open(cat,'rb') as f:
    hdul=fits.open(f)
    print(hdul[1].data)
'''
data=[]
for forced in allforced:
    with open(forced,'rb') as f:
        hdul=fits.open(f)
        #print(hdul[1].header,hdul[1].data)
        data.append([hdul[1].data[0][14],hdul[1].data[0][0],hdul[1].data[0][5],hdul[1].data[0][6]])
        print('mjd = ',hdul[1].data[0][14],'ra = ',hdul[1].data[0][20],'dec = ',hdul[1].data[0][21],'flux = ',hdul[1].data[0][0],'fluxPoint = ',hdul[1].data[0][5],'fluxGal = ',hdul[1].data[0][6])

data=np.asfarray(data)

sorteddata=data[data[:,0].argsort()]

plt.plot(sorteddata[:,0],sorteddata[:,1],label='total flux')
plt.plot(sorteddata[:,0],sorteddata[:,2],label='fluxPoint')
plt.plot(sorteddata[:,0],sorteddata[:,3],label='fluxGal')
plt.legend()
plt.xlabel('mjd')
plt.ylabel('flux (nanomaggies)')
plt.savefig('ZTF18abcfdzu_lc.png')
def mag(x):
    return 22.5-2.5*np.log10(x)
plt.clf()
now=Time.now()
print(now.jd)
plt.plot(sorteddata[:,0]-now.jd,mag(sorteddata[:,1]),label='total flux')
plt.plot(sorteddata[:,0]-now.jd,mag(sorteddata[:,2]),label='fluxPoint')
plt.plot(sorteddata[:,0]-now.jd,mag(sorteddata[:,3]),label='fluxGal')
plt.ylim((22,18.5))
plt.legend()
plt.xlabel('Days Ago')
plt.ylabel('Magnitude')
plt.savefig('ZTF18abcfdzu_lcmag.png')

