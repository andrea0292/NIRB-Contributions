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
from scipy.integrate import quad
from scipy.integrate import fixed_quad
from scipy.integrate import romberg
import PS
import tools 


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
ps1h=read_ps(NIRB_ps_path + "PS_GALl_1h.dat")
ps2h=read_ps(NIRB_ps_path + "PS_GALl_2h.dat")

def PS2(z, k):
    return ps_interp(z ,k , ps1h[2] + ps2h[2], ps1h[0], ps1h[1])


class LowGal:
    def __init__(self):
        """
        Container class to compute various quantities for low-z Galaxy as studied here --> https://arxiv.org/pdf/1201.4398.pdf

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
        self.Mpc = 1e3 * const.kpc.value
        self.zmin = 0.001
        self.zmax = 30.001
        self.zbins = 301
        self.h = cosmo.h
        self.lambda_eff = np.array([0.15,0.36,0.45,0.55,0.65,0.79,0.91,1.27,1.63,2.20,3.60,4.50])
        self.Mstar0_vect = np.array([-19.62,-20.20,-21.35,-22.13,-22.40,-22.80,-22.86,-23.04,-23.41,-22.97,-22.40,-21.84])
        self.q_vect = np.array([1.1,1.0,0.6,0.5,0.5,0.4,0.4,0.4,0.5,0.4,0.2,0.3])
        self.phistar0_vect = np.array([2.43,5.46,3.41,2.42,2.25,2.05,2.55,2.21,1.91,2.74,3.29,3.29])
        self.p_vect = np.array([0.2,0.5,0.4,0.5,0.5,0.4,0.4,0.6,0.8,0.8,0.8,0.8])
        self.alpha0_vect = np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.])
        self.r_vect = np.array([0.086,0.076,0.055,0.060,0.070,0.070,0.060,0.035,0.035,0.035,0.035,0.035])
        self.zmax_vect = np.array([8.0,4.5,4.5,3.6,3.0,3.0,2.9,3.2,3.2,3.8,0.7,0.7]) 



    def fnu(self, m):
        """
        Specific flux in micro Jansky, as in Eq.2 of --> 1201.4398. Function of the magnitude m
        """
        return 10**(-0.4*(m -23.9))

    def Absolute(self, m,z): 
        # Pass from apparent to absolute magnitude, as in Eq. 8. Achtung, the log needs to be in basis 10
        return m - cosmo.distmod(z).value + 2.5 * np.log10(1+z)

    def Ms(self,z,q,Mstarz0):
        return Mstarz0 - 2.5*np.log10((1+(z-0.8))**q) #Eq.3

    def phis(self,z,p,phistar0):
        return phistar0*np.exp(-p*(z-0.8))*1e-3 # Eq.4

    def alphaMe(self,z,r,alpha0):
        return alpha0*(z/0.01)**r # Eq. 5

    def phi_star_highz(self,z):
        return 1.e-3*1.14*10.**(0.003*(z-3.8))

    def Mshighz(self,z):

        return self.Ms(3.8,self.q_vect[0],self.Mstar0_vect[0]) + 0.33*(z-3.8) # modified to match 

    def MsUV(self, z,q,Mstarz0):
        if z > 3.8:
            return self.Mshighz(z)
        else:
            return self.Ms(z,q,Mstarz0)

    def alphahighz(self, z):
        """
        function that computes alpha at high redshift (z>3.8), defined as in Section 7.5 of Bowens et al. 2003, ApJ 737
        """
        return self.alphaMe(3.8,self.r_vect[0],self.alpha0_vect[0]) - 0.018*(z-3.8)


    def Msobs(self, lobs,z):
        lambdaemission = lobs / (1 + z)
        indice = np.argmin(np.abs(lambdaemission - self.lambda_eff))
        if indice==0 and z > 3.8:
            return self.Mshighz(z)
        else:
            return self.Ms(z,q_vect[indice],Mstar0_vect[indice])
       

    def PhiObs(self, lobs, m , z):
        M = self.Absolute(m,z)
        exponent = 0.4 * (self.Msobs(lobs,z) - M )
        return 0.4 * np.log(10) * self.phisobs(lobs,z) * (10**exponent) ** (self.alpha0obs(lobs,z) + 1) * np.exp(-10**exponent)

    def PhiMe(self, z,m,phistar0,p,Mstarz0,q,alpha0,r):
        M = M = self.Absolute(m,z)
        exponent = 0.4 * (self.Ms(z,q,Mstarz0) - M )
        return 0.4*np.log(10)*self.phis(z,p,phistar0)* (10**exponent) ** (self.alphaMe(z,r,alpha0) + 1) * np.exp(-10**exponent)


    def PhiMeHigh(self, z,m):
        M = self.Absolute(m,z)
        exponent = 0.4 * (self.Mshighz(z) - M )
        return 0.4*np.log(10)*self.phi_star_highz(z)* (10**exponent) ** (self.alphahighz(z) + 1) * np.exp(-10**exponent)


    # Now we define the functions with input the observed wavelength (to be passed in micron)

    def phisobs(self,lobs,z):

        lambdaemission = lobs / (1 + z)
        indice = np.argmin(np.abs(lambdaemission - self.lambda_eff))

        if indice ==0 and z > 3.8:
            
            return self.phi_star_highz(z)

        else:
            return self.phis(z,self.p_vect[indice],self.phistar0_vect[indice])

    def PhiObsMe2(self, lambda_obs,m,z):
    
        z_vect = lambda_obs/self.lambda_eff - 1.
        phi_vect = np.zeros(len(z_vect)) + 1.e-30
        ind_gt0 = np.where(z_vect > 0)[0]
        ind_eq0 = np.where(z_vect == 0)[0]
        phi_vect[ind_gt0] = np.array([self.PhiMe(z_vect[i],m,self.phistar0_vect[i],self.p_vect[i],self.Mstar0_vect[i],self.q_vect[i],self.alpha0_vect[i],self.r_vect[i]) for i in ind_gt0])
        phi_vect[ind_eq0] = self.PhiMe(0.01,m,self.phistar0_vect[ind_eq0],self.p_vect[ind_eq0],self.Mstar0_vect[ind_eq0],self.q_vect[ind_eq0],self.alpha0_vect[ind_eq0],self.r_vect[ind_eq0])
        if (z_vect[0] >= 4.):
            phi_vect[0] = self.PhiMeHigh(z_vect[0],m)
        if ((z > z_vect[0]) or (z < z_vect[-1])):
            return 1.e-30
        else:
            ind_low,ind_high = tools.find_nearest(z,z_vect)
        #if ((z_vect[ind_low] <= 0) or (z_vect[ind_high] <=0)) :
        #    return 0.
        #else:     
        return tools.lininterp(z,z_vect[ind_low],z_vect[ind_high],phi_vect[ind_low],phi_vect[ind_high])


    def alpha0obs(self,lobs,z):

        lambdaemission = lobs / (1 + z)
        indice = np.argmin(np.abs(lambdaemission - self.lambda_eff))

        if indice==0 and z > 3.8:
            return self.alpha_highz(z)
        else:
            return self.alphaMe(z,self.r_vect[indice],self.alpha0_vect[indice])

    def Msobs(self, lobs,z):
        lambdaemission = lobs / (1 + z)
        indice = np.argmin(np.abs(lambdaemission - self.lambda_eff))
        if indice==0 and z > 3.8:
            return self.MsUV(z,self.q_vect[0],self.Mstar0_vect[0])
        else:
            return self.Ms(z,self.q_vect[indice],self.Mstar0_vect[indice])

    def LF(self, lobs, m , z): # Luminosity function as in Eq. 6 of the paper 
        M = self.Absolute(m,z)
        exponent = 0.4 * (self.Msobs(lobs,z) - M )
        return 0.4 * np.log(10) * self.phisobs(lobs,z) * (10**exponent) ** (self.alpha0obs(lobs,z) + 1) * np.exp(-10**exponent)

    def E(self, z):
        return cosmo.H(z).value/cosmo.H(0).value

    def dVdz(self, z):
        dH0 = (1e-3 * const.c.value/cosmo.H(0).value)
        return cosmo.comoving_distance(z).value**2 * dH0/self.E(z)

    def Nuz(self, lambdaobs, m, z):
        return self.PhiObsMe2(lambdaobs, m , z) * self.dVdz(z)

    def dFdz(self, lobs, mlim, z): # Returned in nW/m^2/sr, the 10^-32 comes from the micro-Jansky, the 1e-9 from the nano-Watt
        if z > 8:
            return 0
        else:
            def integrand(m):

                return const.c.value / (lobs * 1e-6) * self.Nuz(lobs,m, z) * self.fnu(m) * 1e-32 / 1e-9

        return quad(integrand,mlim,65)[0]

    def WindowGal(self, lobs,mlim,z): # nW/m^2/sr/Mpc ---> H/c dF/dz
        c_km = 1e-3 * const.c.value
        return cosmo.H(z).value/c_km * self.dFdz(lobs, mlim, z)

    def IntermediatePsNoise(self,lobs, mlim, z):
        marray = np.arange(mlim,60,0.5)
        integrand_array= np.array([self.Nuz(lobs, m, z) * (const.c.value / (lobs * 1e-6) * self.fnu(m) * 1e-32 / 1e-9)**2 for m in marray])
        return np.trapz(integrand_array,marray)

    def PSnoise(self, lobs, mlim):

        #z_vec = np.logspace(-3, 0.7, 30)
        z_vec = np.logspace(np.log10(0.001), np.log10(7), 60)
        array_dNf = np.array([self.IntermediatePsNoise(lobs, mlim, z) for z in z_vec])
    
        return np.trapz(array_dNf,z_vec)

class Cllow:
    def __init__(self):
        """
        Container class to compute the Power spectrum and the Cl 

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
        PSgal1h = np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/NIRB_PS/PS_GALl_1h.dat")
        PSgal2h = np.loadtxt("/Users/andreacaputo/Desktop/Phd/AxionDecayCrossCorr/Codes/NIRB_PS/PS_GALl_2h.dat")
        self.Mpc = 1e3 * const.kpc.value
        self.zmin = 0.001
        self.zmax = 30.001
        self.zbins = 301
        self.h = cosmo.h
        self.z_vect = np.linspace(self.zmin, self.zmax, self.zbins)
        self.k_vect = PSgal1h[:,0]* self.h
        self.Power1h = PSgal1h[:,1:]/(self.h**3)
        self.Power2h = PSgal2h[:,1:]/(self.h**3)
        self.Power = self.Power1h + self.Power2h
        self.Praw_prova1h = interp2d(self.k_vect, self.z_vect, np.transpose(self.Power1h))
        self.Praw_prova2h = interp2d(self.k_vect, self.z_vect, np.transpose(self.Power2h))
        self.Praw_prova = interp2d(self.k_vect, self.z_vect, np.transpose(self.Power))


    def PSgal1h(self, k, z):

        return self.Praw_prova1h(k, z)[0]

    def PSgal2h(self, k, z):
        return self.Praw_prova2h(k, z)[0]

    def PSgal(self, k, z):
        return self.Praw_prova(k, z)[0]

    def tointegrate(self, lobs,mlim,k,z):

        lowz = LowGal()
        return (1e-3*const.c.value/cosmo.H(z).value)*self.PSgal(k, z)*(lowz.WindowGal(lobs,mlim,z))**2 / (cosmo.comoving_distance(z).value)**2
    
    def Pqlow(self, lobs,mlim,k):
        def integrand(z):
            return self.to_integrate(lobs,mlim,k,z)
        return quad(integrand,1e-4,7)[0]

    def CllowQuad(self,lobs,mlim,l):
        def integrand(z):
            kl = l/cosmo.comoving_distance(z).value
            return self.tointegrate(lobs,mlim,kl,z)
        return quad(integrand,0.001,7)[0] # ACHTUNG: the lower bound should be consistent with the PS data, otherwise it will interpolate


    def CllowNoise(self,lobs,mlim): # Here the Cl is already given in (nW/m^2/sr)^2
        lowz = LowGal()

        return lowz.PSnoise(lobs, mlim)

    def CllowNn(self,lobs,mlim,l): # Here the Cl is already given in (nW/m^2/sr)^2, no noise
        lowz = LowGal()
        zarray = np.logspace(np.log10(0.001), np.log10(7), 60)
        kl = l/ cosmo.comoving_distance(zarray).value
        array = np.arange(0,len(zarray))
        yarray = np.array([self.tointegrate(lobs,mlim,kl[i],zarray[i]) for i in array])
        return np.trapz(yarray,zarray) 

    def Cllow(self,lobs,mlim,l): # Here the Cl is already given in (nW/m^2/sr)^2
        lowz = LowGal()
        zarray = np.logspace(np.log10(0.001), np.log10(7), 60)
        kl = l/ cosmo.comoving_distance(zarray).value
        array = np.arange(0,len(zarray))
        yarray = np.array([self.tointegrate(lobs,mlim,kl[i],zarray[i]) for i in array])
        return np.trapz(yarray,zarray) + lowz.PSnoise(lobs, mlim) 

