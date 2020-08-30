import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.misc import derivative
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from tqdm import *
from astropy.cosmology import Planck13 as cosmo
from astropy import constants as const
import sys
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import tools
import scipy
from scipy.integrate import quad
from scipy.integrate import fixed_quad
from scipy.integrate import romberg
import PS
import tools

# Here we import some quantities that are useful. Namely the Power Spectra (1halo and 2halos terms) for the galaxies at high z > 6
# and also the star formation rate density (SFRD) and the stellar mass density. See https://arxiv.org/pdf/1205.2316.pdf

PSgal1h = np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/NIRB_PS/PS_GALh_1h.dat")
PSgal2h = np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/NIRB_PS/PS_GALh_2h.dat")

Mpc = 1e3 * const.kpc.value
zmin = 0.001
zmax = 30.001
zbins = 301
h = cosmo.h
z_vect = np.linspace(zmin, zmax, zbins)
k_vect = PSgal1h[:,0]*h
Power1h = PSgal1h[:,1:]/h**3
Power2h = PSgal2h[:,1:]/h**3
Power = Power1h + Power2h
Praw_prova1h = interp2d(k_vect, z_vect, np.transpose(Power1h))
Praw_prova2h = interp2d(k_vect, z_vect, np.transpose(Power2h))
Praw_prova = interp2d(k_vect, z_vect, np.transpose(Power))

# Here I interpolate with the usual Python routines

def PSgal(k, z):
    return Praw_prova(k, z)

def PSgal1h(k, z):
    return Praw_prova1h(k, z)

def PSgal2h(k, z):
    return Praw_prova2h(k, z)

zzmin = 0.001
zzmax = 30.001
zzbins = 301

def read_ps(filename):
    h = cosmo.h
    tbl = np.loadtxt(filename)
    z_vect = np.linspace(zzmin, zzmax, zzbins)
    k_vect = tbl[:,0]
    matrix = tbl[:,1:]
    return z_vect, k_vect*h, matrix/(h**3)

# Here I also check with my own function 

def ps_interp(z,k,matrix,z_vect,k_vect):
    if (z < np.min(z_vect) or (z > np.max(z_vect)) or (k < np.min(k_vect)) or (k > np.max(k_vect)) ):
        return 0.
    ind_z_low, ind_z_up = tools.find_nearest(z_vect,z)
    ind_k_low, ind_k_up = tools.find_nearest(k_vect,k)
    z_low = z_vect[ind_z_low]
    z_up = z_vect[ind_z_up]
    k_low = k_vect[ind_k_low]
    k_up = k_vect[ind_k_up]
    if ((matrix[ind_k_low, ind_z_low] <= 0.) or (matrix[ind_k_up, ind_z_low] <=0 ) or (matrix[ind_k_low, ind_z_up] <=0) or (matrix[ind_k_up, ind_z_up] <=0)):
        return 0.
    else:
        ps_z_low = 10.**tools.lininterp(k, k_low, k_up, np.log10(matrix[ind_k_low, ind_z_low]), np.log10(matrix[ind_k_up,ind_z_low]))
        ps_z_up = 10.**tools.lininterp(k, k_low, k_up,np.log10(matrix[ind_k_low, ind_z_up]), np.log10(matrix[ind_k_up,ind_z_up]))
    return tools.lininterp(z,z_low,z_up,ps_z_low,ps_z_up)

NIRB_ps_path = "./NIRB_PS/"   
ps1h=read_ps(NIRB_ps_path + "PS_GALh_1h.dat")
ps2h=read_ps(NIRB_ps_path + "PS_GALh_2h.dat")

def PS2(z, k):
    return ps_interp(z ,k , ps1h[2] + ps2h[2], ps1h[0], ps1h[1])

arrayMII = np.logspace(np.log10(3), np.log10(150), 100)
arrayMIII = np.logspace(np.log10(3), np.log10(500), 100)

# Here I ulpoad the quantity calculated with the Fortran code --> https://wwwmpa.mpa-garching.mpg.de/~komatsu/CRL/massfunction/collapsefractionst/fcoll_shethtormen/



#IntzdndM=np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/Int_z_dndM2.dat")
IntzdndM=np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/Int_z_dndM2_Mmin6.dat")
zarray= IntzdndM[:,0]
intarray = IntzdndM[:,1]

toint= interp1d(zarray,intarray)
def integranddndM(z):
    return 1.*toint(z)

zarrayC = np.arange(6,31,0.1)
tauImported=np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/tausfC_array_Mmin6.dat")
tointta= interp1d(zarrayC,tauImported)
    
def tausfC(z):
    return 1.*tointta(z)

Data2=np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/tSF_vect_st_Mmin=he6.dat")# change depending on the minimum mass
#AndreaData=np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/tSF_vect_st_Mmin=5e7.dat")# change depending on the minimum mass
z2_psi= Data2[:,0]
rho2_array = Data2[:,1]
tSF2_array = Data2[:,3]
ps2_ary = Data2[:,2]

tointpsi2= interp1d(z2_psi,ps2_ary)
def psi2(z):
    return 1.*tointpsi2(z)


tointSF= interp1d(z2_psi,tSF2_array)
year = 60*60*24*365
def tausfC2(z):
    return 1.*tointSF(z)*year

tointrho2= interp1d(z2_psi,rho2_array)
def rho2(z):
    return 1.*tointrho2(z)

# Here instead I calculate the same quantities using Colossus. The agrement between the two codes is pretty good.

from colossus.cosmology import cosmology
cosmology.setCosmology('planck15')
from colossus.lss import mass_function

def MfunctionColossusLog(m, z):
    return cosmo.h**3 * mass_function.massFunction(m*cosmo.h,z,q_in='M', model = 'sheth99', q_out = 'dndlnM')


def integralColossusM2(z, Mmin):
    marray = 10**np.arange(Mmin,20, 0.05)
    to_integrate = np.array([m*MfunctionColossusLog(m, z) for m in marray])
    return np.trapz(to_integrate, marray)


class StarProperties:
    def __init__(self):
        """
        Container class to compute various contributions for high-z Galaxy, see https://arxiv.org/pdf/1205.2316.pdf

        """

        # Set constants
        self.fromHztoeV = 6.58e-16
        self.gramstoeV = 1 / ( 1.78 * 1e-33)
        self.mtoev = 1/(1.97 * 1e-7) 
        self.H0 = cosmo.H(0).value * 1e3 / (1e3 * const.kpc.value) #expressed in 1/s
        self.rhocritical = cosmo.critical_density(0).value * self.gramstoeV /(1e-2)**3 # eV/m**3
        self.Om0 = cosmo.Om0 #total matter 
        self.OLambda0 = cosmo.Ode0  # cosmological constant
        self.DM0 = self.Om0 - cosmo.Ob0 # dark matter
        self.evtonJoule = 1.60218 * 1e-10 # from eV to nJ
        self.evtoJoule = 1.60218 * 1e-19 # from eV to J
        self.h  = 0.6766
        PS1h = np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/NIRB_PS/PS_DM_1h.dat")
        PS2h = np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/NIRB_PS/PS_DM_2h.dat")
        self.Mpc = 1e3 * const.kpc.value
        self.zmin = 0.001
        self.zmax = 30.001
        self.zbins = 301
        self.h = cosmo.h
        self.TotPower = PS1h + PS2h
        self.z_vect = np.linspace(self.zmin, self.zmax, self.zbins)
        self.k_vect = self.TotPower[:,0]*self.h
        self.Power = self.TotPower[:,1:]/self.h**3
        self.Praw_prova  = interp2d(self.k_vect, self.z_vect, np.transpose(self.Power))
        self.Msun = const.M_sun.value #Kg
        self.Lsun = const.L_sun.value # in W
        self.Sb = const.sigma_sb.value  # Stefan-Boltzmann constant in  W/m**2/K**4
        self.Rsun = const.R_sun.value # in meters
        self.hplanck = const.h.value # Js
        self.cc = const.c.value #m/s
        self.kb = const.k_B.value # kb constant J/K
        self.mp = const.m_p.value # proton mass in Kg
        self.Hseconds= cosmo.H(0).value / (1e6 * u.parsec.to(u.km)) # in 1/s
        self.nHnebula = 1e4 / (1e-2)**3 # in m**-3, is also equal to the electron number density
        self.eVtoK = 1.16e4 # from eV to K
        self.year = 3.154e+7 # from year to seconds
        self.Ry = 1.1e7 #1/m Rydberg constant
        self.ergtoJoule = 1e-7 # from erg to Joule
        self.fromJouletoeV = 6.242e+18 # from Joule to eV 
        self.RyK = 13.6057 * self.eVtoK  # Rydber constant in eV, then converted in K
        self.Mpc = 1e3 * const.kpc.value 
        self.eVtoK = 1.16e4 # from eV to K
        self.nuly = 2.4663495e15 # Lyman alpha frequency in Hz, which corresponds to a photon energy of 10.2 eV

    # Here we define the stellar initial mass functions (IMF) we adopt two descriptions,
    # Salpeter mass function (pop II) and the Larson one (pop III)

    def Salpeter_non_norm(self, Mstar):
        
        return Mstar**(-2.35)
    
    def Salpeter(self,Mstar):
        def toint(Mstar):
            return self.Salpeter_non_norm(Mstar)
        normalization = quad(toint,3,150)[0] #quad(Salpeter_non_norm,1e-15,np.inf)[0]  ?? which range? 
        return self.Salpeter_non_norm(Mstar) / normalization

    def Larson_non_norm(self,Mstar):
        
        Mcritical = 250 
        return Mstar**(-1) * (1 + Mstar/Mcritical)**(-1.35)
    
    def Larson(self,Mstar):
        normalization = quad(self.Larson_non_norm,3,500)[0] # quad(Larson_non_norm,1e-15,np.inf)[0]
        Mcritical = 250 
        return self.Larson_non_norm(Mstar) / normalization

    # Now we define various quantities, namely the bolometric luminosity, the effective
    # temperature, the main sequence life time and the hydrogen photoionization rate
    # First for PopII (Salpeter Mass)
    def LbolII(self,Mstar):
        x = np.log10(Mstar/1)
        exponent = 0.138 + 4.28 * x - 0.653 * x**2
        return self.Lsun * 10**exponent # in Sun Luminosity

    def TeffII(self,Mstar):
        x = np.log10(Mstar/1)
        exponent = 3.92 + 0.704 * x - 0.138 * x**2
        return 10**exponent # in Kelvin

    def tauII(self, Mstar):
        x = np.log10(Mstar/1)
        exponent = 9.59 -2.79*x + 0.63 * x**2
        return 10**exponent * self.year # in seconds

    def QbarII(self, Mstar):
        x = np.log10(Mstar/1)
        exponent = 27.80 + 30.68*x -14.80*x**2 +2.50 * x**3
        return 10**exponent # rate in s**-1

    # Now for the Pop III, the Larson model

    def LbolIII(self, Mstar):
        x = np.log10(Mstar/1)
        exponent = 0.4568 + 3.897 * x - 0.5297 * x**2
        return self.Lsun * 10**exponent # in Sun Luminosity

    def TeffIII(self, Mstar):
        x = np.log10(Mstar/1)
        exponent = 3.639 + 1.501 * x -0.5561 * x**2 + 0.07005 * x**3
        return 10**exponent # in Kelvin

    def tauIII(self, Mstar):
        x = np.log10(Mstar)
        exponent = 9.785 - 3.759*x + 1.413 * x**2 - 0.186 * x**3
        return 10**exponent * self.year # in seconds

    def QbarIII(self, Mstar):
        x = np.log10(Mstar/1)
        if 5 <= Mstar <= 9:
            exponent = 39.29 +8.55 *x
            return 10**exponent
        if 9 < Mstar < 500:
            exponent = 43.61 +4.90*x -0.83*x**2
            return 10** exponent
        else:
            return 0
    # Now we define the stellar radius, which enters the stellar emission spectrum

    def RadiusII(self, Mstar):
        return np.sqrt(self.LbolII(Mstar)/self.Sb/self.TeffII(Mstar)**4/4/np.pi) # in meter

    def RadiusIII(self, Mstar):
        return np.sqrt(self.LbolIII(Mstar)/self.Sb/self.TeffIII(Mstar)**4/4/np.pi) # in meter

    def tauaverage(self,whichpop):
        if whichpop == 'II':
            def tointegrate(M):
                return self.tauII(M)*self.Salpeter(M)
            return quad(tointegrate,3,150)[0]
        if whichpop == 'III':
            def tointegrate(M):
                return self.tauIII(M)*self.Larson(M)
            return quad(tointegrate,3,500)[0]

    # First we define the average masses for the two populations
    def MassAverage(self, whichpop):
        if whichpop == 'II':
            def to_integrate(Ms):
                return Ms * self.Salpeter(Ms)
            return quad(to_integrate, 3,150)[0]
    
        if whichpop == 'III':
            def to_integrate(Ms):
                return Ms * self.Larson(Ms)
            return quad(to_integrate, 3,500)[0]

    def tauaverage(self,whichpo):

        if whichpo == 'II':
            def to_integrate(m):
                return self.tauII(m)*self.Salpeter(m)
            return quad(to_integrate,3,150)[0]
        if whichpo == 'III':
            def to_integrate(m):
                return self.tauIII(m)*self.Larson(m)
            return quad(to_integrate,3,500)[0]


    def Stellar(self, Mstar, population,nu):

        #fromJouletoeV = 6.242e+18
        heV=4.136e-15 # Planck's constant [eV s]
    
        if heV*nu < 13.6:
    
            def PlanckSpectrum(T):
                denominator = np.exp(self.hplanck*nu/self.kb/T) - 1
                return 2*self.hplanck*nu**3/self.cc**2 / denominator

            if population == 'II':
                Sstar = 4*np.pi * self.RadiusII(Mstar)**2
                Lumi = np.pi * Sstar * PlanckSpectrum(self.TeffII(Mstar))
            
            if population == 'III':
                Sstar = 4*np.pi * self.RadiusIII(Mstar)**2
                Lumi = np.pi * Sstar * PlanckSpectrum(self.TeffIII(Mstar))
            return Lumi
    
        else:
            return 0

    # Eq.8 of the Cooray paper, is the fitting formula for the fraction of Lya photons produced
    # in the case of B recombination


    # Temperature should be given in K
    def frec(self,T):
        return 0.686 -0.106*np.log10(T/1e4) - 0.009*(T/1e4)**(-0.44)

    # Now we define the Hydrogen B recombination coefficient. T in Kelvin

    def alphaBrec(self,T):
        T4 = T/1e4
        a = 4.309
        b= -0.6166
        c= 0.6703
        d = 0.53
        return 1e-13 * a *T4**b / ( 1 + c * T4**d) *(1e-2)**3 # Units m**3 / s, converted from the paper  where it was in cm**3/s

    def epsilonrec(self,T,z):
        
        return self.alphaBrec(T) * self.frec(T) * self.nHnebula**2
    
    def Coeff(self, T, whichtransition):
    
        gl = 1 # statistical weight, is one if we consider the hydrogen in its  fundamental level
    
        if whichtransition == '2p':
            def gamma2p(T):
                if 5000 <= T <=55000:
                    return 3.435e-1 + 1.297e-5 * T + 2.178e-12*T**2 + 7.928e-17 * T**3
                if 55000 <= T <= 5e5:
                    return 3.162e-1 + 1.472e-5 * T - 8.275e-12*T**2 - 8.794e-19 * T**3
            
            E2p = 10.2 * self.eVtoK 
            return 8.629e-6 /(gl)/np.sqrt(T) * gamma2p(T) * np.exp(-E2p/T) *(1e-2)**3 # m**3/s
    
        if whichtransition == '3s':
        
            def gamma3s(T):
            
                if 5000 <= T <=55000:
                    return 6.250e-2 - 1.299e-6 * T + 2.666e-11*T**2 - 1.596e-16 * T**3
                if 55000 <= T <= 5e5:
                    return 3.337e-2 + 2.223e-7 * T - 2.794e-13*T**2 + 1.516e-19 * T**3
            
            E3s = 12.0873 * self.eVtoK 
            return 8.629e-6 /(gl)/np.sqrt(T) * gamma3s(T) * np.exp(-E3s/T) *(1e-2)**3 # m**3/s
    
        if whichtransition == '3d':
            def gamma3d(T):
            
                if 5000 <= T <=55000:
                    return 5.030e-2 + 7.514e-7 * T - 2.826e-13*T**2 - 1.098e-17 * T**3
                if 55000 <= T <= 5e5:
                    return 5.051e-2 + 7.876e-7 * T - 2.072e-12*T**2 + 1.902e-18 * T**3
        
            E3d = 12.0873 * self.eVtoK 
            return 8.629e-6 /(gl)/np.sqrt(T) * gamma3d(T) * np.exp(-E3d/T) *(1e-2)**3 # m**3/s

    def epsiloncoll(self,T, z): # Eq.10
        Ceff = self.Coeff(T, '2p') + self.Coeff(T, '3s') + self.Coeff(T, '3d')

        zarray = np.linspace(6,30)
        H1imported=np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/nH1s_zlinspace6_30.dat")
        tointH1= interp1d(zarray,H1imported)
        def nH1s(z):
            return 1.*tointH1(z)
        
        return Ceff * self.nHnebula*nH1s(z) # not sure if in the nebula there is nHI, the paper doesnt' mention it, I have pute nH1s(z)

    def E(self,z):
        return cosmo.H(z).value/cosmo.H(0).value

    def phiprofile(self, nu, z):
        omegab = 0.0482754208891869
        if self.nuly - nu <= 0:
            return 0
        else:
            dnu = self.nuly - nu #nu star in Hz
            nustar = 1.5e11*(omegab*cosmo.h**2/0.019)*(cosmo.h/0.7)**(-1)*(1+z)**3 /self.E(z)
        return nustar * dnu**(-2) * np.exp(-nustar/dnu)

    # Now we write the ionization volumes as in Eq.4-5, we give them in m**3, as we are trying to work in SI units

    def Vneb(self, T, Mstar, whichpop):
     
        if whichpop == 'II':
            return self.QbarII(Mstar)/(self.nHnebula**2 * self.alphaBrec(T))
    
        if whichpop == 'III':
            return self.QbarIII(Mstar)/(self.nHnebula**2 * self.alphaBrec(T))

    def Vol(self, T, Mstar, whichpop,z,where):
    
        if where == 'nebula':
            return self.Vneb(T, Mstar, whichpop)
        if where == 'IGM':
            return self.VIGM(Mstar, whichpop,z)

    # Final luminosity due to Lyalpha (in Joule)
    def LumiLy(self,whichpop, T, z, nu, Mstar):

        if whichpop == 'II':
            Q = self.QbarII(Mstar)
        if whichpop == 'III':
            Q = self.QbarIII(Mstar)
    
        Volume = self.Vneb(T, Mstar, whichpop)
    
        return self.hplanck * self.nuly * 0.64 * self.phiprofile(nu, z) * Q

    def findn(self, nu): # frequency passed in 1/s (Hz)
        toreturn = 2
        for i in range(2,100):
            if self.cc * self.Ry/ i**2 < nu < self.cc * self.Ry/ (i - 1)**2:
                toreturn = i
            else:
                toreturn = toreturn
        return toreturn

    def specificemission(self,nu, T, z, whichprocess): # ff and fb processes
    
        nlevel = self.findn(nu)
        
        xn = self.RyK / T / (nlevel**2) 
    
        
        nenp = (self.nHnebula/1e6)**2 #converted to cm for the moment
        
        if whichprocess == 'ff':
            gaunt = 1.1
        if whichprocess == 'fb':
            gaunt = xn*np.exp(xn) * 1.05 / nlevel
        
        return 5.44e-39 * np.exp(-self.hplanck *nu/self.kb/T) / np.sqrt(T) * nenp * gaunt * self.ergtoJoule /(1e-2)**3 # Joule/m**3 / s/ Hz /sr

    # this is the luminosity associated with ff and fb processe 

    def Lumifffb(self, nu, T, z, Mstar, whichprocess, whichpop):
    
        return 4*np.pi * self.specificemission(nu, T, z, whichprocess) * self.Vneb(T, Mstar, whichpop)

    #normalized probability of generating one photon, fitted formula
    def Pone(self, y):
        return 1.307 -2.627*(y-0.5)**2 +2.563*(y-0.5)**4 - 51.69*(y-0.5)**6

    def Lumi2ph(self, nu, T,z, Mstar,whichpop): # Again in Joule


        if whichpop == 'II':
            Q = self.QbarII(Mstar)
        if whichpop == 'III':
            Q = self.QbarIII(Mstar)
        if self.nuly - nu <= 0:
            return 0
        else:
            ratio = nu/ self.nuly
        return 2*self.hplanck*ratio*self.Pone(ratio)*(1-0.64)*Q#*epsilon2ph(T, where,z)*Vol(T, Mstar, whichpop,z,where)

    # Now we define a function that pick the process you want to consider for the luminosity 

    def PickLumi(self, whichp, Mstar,nu,T, z, whichpop): # process, mass, frequency, temperature, redshift, stellar population 
        if whichp == 'stellar':
            return self.Stellar(Mstar, whichpop,nu)
        if whichp == 'Ly':
            return self.LumiLy(whichpop, T, z, nu, Mstar)
        if whichp == 'ff':
            return self.Lumifffb(nu, T, z, Mstar,  'ff', whichpop)
        if whichp == 'fb':
            return self.Lumifffb(nu, T, z, Mstar, 'fb', whichpop)
        if whichp == '2p':
            return self.Lumi2ph(nu, T,z, Mstar, whichpop)
        else:
            print('You have to select: stellar, Ly, ff, fb or 2p')

    # We consider only the Nebula contribution at the end of the day, IGM is negligible

    def lnu(self, whichp, nu,T, z, whichpop,who):

        tauavII= self.tauaverage('II')
        averageII = self.MassAverage('II')
        tauavIII= self.tauaverage('III')
        averageIII = self.MassAverage('III')

        # Here I distinguish between the two output from Komatsu's code and Colossus
    
        if who == 'Colossus':
            def tSFtouse(z):
                return tausfC(z)
        if who == 'Komatsu':
            def tSFtouse(z):
                return tausfC2(z)

        if whichpop == 'II':
        
            #arrayM = np.arange(3,150,0.1)

            arrayM = arrayMII
        
            def auxiliary(mass):
            
                if tauavII > tSFtouse(z):#tauII(mass)> tausfCint(z):#tausf(z):
    
                    return  self.Salpeter(mass)*self.PickLumi(whichp,mass,nu,T, z, whichpop)/averageII
                else:
                    return self.tauII(mass)*self.Salpeter(mass)*self.PickLumi(whichp,mass,nu,T, z, whichpop)/(averageII*tSFtouse(z))
            
            arrayL = np.array([auxiliary(m) for m in arrayMII])
   
        if whichpop == 'III':
        
            arrayM = arrayMIII
        
            def auxiliary(mass):
            
                if tauavIII > tSFtouse(z):  #tauIII(mass)> tausfCint(z):
                    return  self.Larson(mass)*self.PickLumi(whichp,mass,nu,T, z, whichpop)/averageIII
                else:
                    return self.tauIII(mass)*self.Larson(mass)*self.PickLumi(whichp,mass,nu,T, z, whichpop)/(averageIII*tSFtouse(z))
            
            arrayL = np.array([auxiliary(m) for m in arrayMIII])
          
        return np.trapz(arrayL, arrayM)
        

    def LnNeb_IItot(self, nu, T, z, fesc, who):

        stel = self.lnu('stellar', nu , T, z, 'II', who) 
                        
        Lyb = (1-fesc)*self.lnu('Ly', nu , T, z, 'II',who)  
    
        fb = (1-fesc)*self.lnu('fb', nu , T, z, 'II',who) 
    
        ff = (1-fesc)*self.lnu('ff', nu , T, z, 'II',who) 
    
        twop = (1-fesc)*self.lnu('2p', nu ,T, z, 'II',who)
        
        return stel + Lyb + fb + ff + twop

    def LnNeb_IIItot(self, nu, T, z, fesc, who):

        stel = self.lnu('stellar', nu , T, z, 'III', who)
                        
        Lyb = (1-fesc)*self.lnu('Ly', nu , T, z, 'III', who)  
    
        fb = (1-fesc)*self.lnu('fb', nu , T, z, 'III', who) 
    
        ff = (1-fesc)*self.lnu('ff', nu , T, z, 'III', who) 
    
        twop = (1-fesc)*self.lnu('2p', nu ,T, z, 'III', who)
        
        return stel + Lyb + fb + ff + twop


    def LnNeb_single(self, nu,T, z, whichpop, fesc):
    
        stel = self.lnu('stellar', nu , T, z,  whichpop)
                        
        Lyb = (1-fesc)*self.lnu('Ly', nu , T, z, whichpop)  
    
        fb = (1-fesc)*self.lnu('fb', nu , T, z,  whichpop) 
    
        ff = (1-fesc)*self.lnu('ff', nu , T, z, whichpop) 
    
        twop = (1-fesc)*self.lnu('2p', nu ,T, z, whichpop)
        
        return stel + Lyb + fb + ff + twop

    def JnuPiecesII(self, T, z, nu,fesc): # We also define the quantity J, from Cooray definitions Eq.32 of https://arxiv.org/pdf/1205.2316.pdf

        tauavII= self.tauaverage('II')
        
        llnu = self.LnNeb_IItot(nu,T, z, fesc, 'Komatsu')

        return 1/(4*np.pi) * llnu * psi2(z) * tauavII/year

    def JnuPiecesIII(self, T, z, nu,fesc): # and for population III
        tauavIII= self.tauaverage('III')
        llnu = self.LnNeb_IIItot(nu,T, z, fesc, 'Komatsu')
        return 1/(4*np.pi) * llnu * psi2(z) * tauavIII/year


    def JnuNebC(self, T, z, nu,fesc): # Eq.31
        sigma_p = 0.5
        fp = 0.5*(1.+scipy.special.erf((z - 10.)/sigma_p))
    
        return  ((1-fp)*self.JnuPiecesII(T, z, nu,fesc) + fp*self.JnuPiecesIII(T, z, nu,fesc))


    def LnNeb_tot(self,nu,T, z, fesc, who):


        sigma_p = 0.5
        fp = 0.5*(1.+scipy.special.erf((z - 10.)/sigma_p))


        return (1-fp)* self.LnNeb_IItot(nu, T, z, fesc, who) + fp*self.LnNeb_IIItot(nu, T, z, fesc, who)

    
        #return 0.5*self.LnNeb_IItot(nu, T, z, fesc, who) + 0.5*self.LnNeb_IIItot(nu, T, z, fesc, who)

class ClHighz:
    def __init__(self):
        """
        Container class to compute the Cl due to high-z contributions; we first define the window functions 

        """

        # Set constants
        self.fromHztoeV = 6.58e-16
        self.gramstoeV = 1 / ( 1.78 * 1e-33)
        self.mtoev = 1/(1.97 * 1e-7) 
        self.H0 = cosmo.H(0).value * 1e3 / (1e3 * const.kpc.value) #expressed in 1/s
        self.rhocritical = cosmo.critical_density(0).value * self.gramstoeV /(1e-2)**3 # eV/m**3
        self.Om0 = cosmo.Om0 #total matter 
        self.OLambda0 = cosmo.Ode0  # cosmological constant
        self.DM0 = self.Om0 - cosmo.Ob0 # dark matter
        self.evtonJoule = 1.60218 * 1e-10 # from eV to nJ
        self.evtoJoule = 1.60218 * 1e-19 # from eV to J
        self.h  = 0.6766
        PS1h = np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/NIRB_PS/PS_DM_1h.dat")
        PS2h = np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/NIRB_PS/PS_DM_2h.dat")
        self.Mpctometer = 3.086e+22 # from Mpc to meters
        self.Mpc = 1e3 * const.kpc.value
        self.zmin = 0.001
        self.zmax = 30.001
        self.zbins = 301
        self.h = cosmo.h
        self.TotPower = PS1h + PS2h
        self.z_vect = np.linspace(self.zmin, self.zmax, self.zbins)
        self.k_vect = PS1h[:,0]*self.h
        self.Power = self.TotPower[:,1:]/self.h**3
        self.Praw_prova  = interp2d(self.k_vect, self.z_vect, np.transpose(self.Power))
        self.Msun = const.M_sun.value #Kg
        self.Lsun = const.L_sun.value # in W
        self.Sb = const.sigma_sb.value  # Stefan-Boltzmann constant in  W/m**2/K**4
        self.Rsun = const.R_sun.value # in meters
        self.hplanck = const.h.value # Js
        self.cc = const.c.value #m/s
        self.kb = const.k_B.value # kb constant J/K
        self.mp = const.m_p.value # proton mass in Kg
        self.Hseconds= cosmo.H(0).value / (1e6 * u.parsec.to(u.km)) # in 1/s
        self.nHnebula = 1e4 / (1e-2)**3 # in m**-3, is also equal to the electron number density
        self.eVtoK = 1.16e4 # from eV to K
        self.year = 3.154e+7 # from year to seconds
        self.Ry = 1.1e7 #1/m Rydberg constant
        self.ergtoJoule = 1e-7 # from erg to Joule
        self.fromJouletoeV = 6.242e+18 # from Joule to eV 
        self.Mpc = 1e3 * const.kpc.value 
        self.RyK = 13.6057 * self.eVtoK
        self.Tgas = 3e4 # Kelvin, below Eq.9 in the paper, said to be used to calculate alphaB
        self.nuly = 2.4663495e15 # Lyman alpha frequency in Hz, which corresponds to a photon energy of
    # 10.2 eV

    # Here we use the definition of Eq.47 of https://arxiv.org/pdf/1205.2316.pdf for the Cl; actually the definition may be a little bit different
    # in the literature but we follow Cooray. We also stick to Komatsu's code at the end of the day. One can easily change to Colossus. It 
    # would be sufficient in the J function to substitute 'Komatsu' --> 'Colossus'
    
    def WindowHighzCooray(self, T,z,fesc,nuobs):

        lnuClass = StarProperties()
        
        nuemission = nuobs * (1+z)
        jnu = lnuClass.JnuNebC(T, z, nuemission,fesc)

        return 1e9/(self.Mpctometer)**2 * nuemission * jnu / (1+z)**2

    def to_integrateCooray(self,nuobs,k,z, T,fesc): 


        factor = (1e-3 * const.c.value/cosmo.H(z).value) *(PSgal(k, z)) / (cosmo.comoving_distance(z).value)**2
    
        return factor * self.WindowHighzCooray(T,z,fesc,nuobs)**2

    def ClHighCooray(self, nuobs,l, T,fesc): #zmin,zmax,N

        arrayz = np.logspace(np.log10(6), np.log10(30), 60)  
        arraytoint = np.array([self.to_integrateCooray(nuobs,l/cosmo.comoving_distance(z).value,z, T,fesc)[0] \
                           for z in arrayz])

        return np.trapz(arraytoint, arrayz) # (nW/m^2/sr)**2