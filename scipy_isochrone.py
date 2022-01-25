import sqlutilpy as sqlutil
import scipy.interpolate 
import matplotlib.pyplot 
import numpy as np
#from idlplotInd import plot
import matplotlib.pyplot as plt
betw = lambda x, x1, x2: (x >= x1) & (x < x2)
mini, gmag = sqlutil.get('''select mini,gmag from isochrones.gaia_1804 
        where age=12000000000 and feh=-2.0;''',
                         host='wsdb.hpc1.cs.cmu.edu')
import imf as iimf
chab = iimf.Chabrier()
mass = chab.distr.rvs(int(1e7))
mini[113] = mini[113] + 1e-10
II = scipy.interpolate.UnivariateSpline(mini, gmag, s=0, k=1, ext=1)
predg = II(mass)
xg = predg[predg != 0]
dmgrid = np.arange(10, 25, 0.1)
ret = np.zeros(len(dmgrid))+0.1
for i in range(len(dmgrid)):
    ret[i] = betw(xg + dmgrid[i], 17, 21).sum()
#plot(dmgrid, ret, ylog=True, yr=[1, 10000000], xtitle='dm', ytitle='Nstars')
plt.semilogy(dmgrid,ret)
plt.xlabel('dm')
plt.ylabel('Nstars')
plt.savefig('/tmp/nstars.pdf')
