import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint
from pynverse import inversefunc
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm

cosmo = FlatLambdaCDM(H0=73,Om0=0.3)
km_Mpc = 1.e3/3.086e22

# Computing dL uncertainty:
def Err(z,dL): return ( 0.1618*z - 0.0289*z**2 + 0.002*z**3 )*dL

# Computing Comoving Volume per Redshift:
def dVc_dz(z):
	dL = cosmo.luminosity_distance(z).value
	H = cosmo.H(z).value*km_Mpc
	return 4*np.pi*dL**2/((1.+z)**2*H)

z_star = 1 # Redshift of the Maximum Distribution
phi0, alpha, beta = 0.015, 2.7, 5.6
C = (z_star+1)*pow(beta/alpha-1, 1/beta) # C = 2.9 in (arXiv:1403.0007)	
def SFR(z): return phi0*(1+z)**alpha/(1+((1+z)/C)**beta) # Star Formation Rate
def Auxiliar(z): return dVc_dz(z)*SFR(z)/(1+z)

# Computing the Normalization Constant
Ntot = 100 ; z_max = 2	# Total number of sources and maximum redshift
n, _ = quad(Auxiliar, 0, z_max)
n /= Ntot

# Defining Our Data Distribution Function
def Pz(z): return Auxiliar(z)/n

def Nz(z): # N(z)
	x,_ = quad(Pz, 0, z)
	return x

N = np.linspace(1,Ntot,Ntot)
z = np.zeros(Ntot)
for i in tqdm(range(Ntot)): z[i] = inversefunc(Nz , N[i], domain=[1.e-10,3])

dL = cosmo.luminosity_distance(z).value
Error = Err(z,dL)
data = np.random.normal( dL, Error )

data/=1.e3 ; Error/=1.e3 ; dL/=1.e3
fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.errorbar( z, data, Error, fmt='k.', capsize=3, elinewidth=0.5, alpha=0.7,label='data')
z = np.linspace(0,2,100); dL = cosmo.luminosity_distance(z).value/1.e3; Error = Err(z,dL)
plt.fill_between(z,dL+Error,dL-Error,color='blue',alpha=0.2,label='$1\sigma$')
plt.fill_between(z,dL+2*Error,dL-2*Error,color='darkblue',alpha=0.1,label='$2\sigma$')
plt.plot(z,dL,'r--',label='$\Lambda CDM$'); plt.legend(loc='best')
plt.xlabel('Redshift'); plt.ylabel('dL [Gpc]')

plt.subplot(1,2,2)
plt.plot(z,Pz(z),'r-')
plt.xlabel('redshift'); plt.ylabel('$dN/dz$')
fig.savefig('simple_p.png')
plt.show()

