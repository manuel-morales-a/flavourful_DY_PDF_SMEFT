"""
How to run the script:
python _muc_neutrinos_NC.py --var1 nu_u --var2 21 --var3 12 --var4 5 --var5 0

Parameters
--var1 : process (i.e nu_u)
--var2 : num_bins_x (i.e 21)
--var3 : num_bins_Q2 (i.e 12)
--var4 : num_bins_E (i.e 5)
--var5 : replica (i.e 0 for the central, 1 for the first replica, etc.)

Process List

- Neutrino Interactions
'nu_u'   : dsigmah_dxdQ2_NC_nu_u,
'nu_ub'  : dsigmah_dxdQ2_NC_nu_ub,
'nu_d'   : dsigmah_dxdQ2_NC_nu_d,
'nu_db'  : dsigmah_dxdQ2_NC_nu_db,
'nu_s'   : dsigmah_dxdQ2_NC_nu_s,
'nu_sb'  : dsigmah_dxdQ2_NC_nu_sb,
'nu_c'   : dsigmah_dxdQ2_NC_nu_c,
'nu_cb'  : dsigmah_dxdQ2_NC_nu_cb,
'nu_b'   : dsigmah_dxdQ2_NC_nu_b,
'nu_bb'  : dsigmah_dxdQ2_NC_nu_bb,

- Antineutrino Interactions
'nub_u'   : dsigmah_dxdQ2_NC_nub_u,
'nub_ub'  : dsigmah_dxdQ2_NC_nub_ub,
'nub_d'   : dsigmah_dxdQ2_NC_nub_d,
'nub_db'  : dsigmah_dxdQ2_NC_nub_db,
'nub_s'   : dsigmah_dxdQ2_NC_nub_s,
'nub_sb'  : dsigmah_dxdQ2_NC_nub_sb,
'nub_c'   : dsigmah_dxdQ2_NC_nub_c,
'nub_cb'  : dsigmah_dxdQ2_NC_nub_cb,
'nub_b'   : dsigmah_dxdQ2_NC_nub_b,
'nub_bb'  : dsigmah_dxdQ2_NC_nub_bb,
"""

#--------------------------------------- Libraries ----------------------------------#
import vegas  
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import numpy.ma as ma
import pandas as pd
from matplotlib.colors import LogNorm
import sys
import os
#------------------------------------ Subprocess Parameters ---------------------------------#
import argparse

parser = argparse.ArgumentParser()

# Aggiungi variabili/argomenti
parser.add_argument('--var1', type=str, help='process', required=True)
parser.add_argument('--var2', type=int, help='num_bins_x', required=True)
parser.add_argument('--var3', type=int, help='num_bins_Q2', required=True)
parser.add_argument('--var4', type=int, help='num_bins_E', required=True)
parser.add_argument('--var5', type=int, help='replica', required=True)
args = parser.parse_args()

process        = args.var1
num_bins_x     = args.var2
num_bins_Q2    = args.var3
num_bins_E     = args.var4
replica        = args.var5



######################################## FILL THIS PART IN THE SAME WAY IN _CENTRAL AND _REPLICA EVERY TIME! ############################################################################################################################################################################################################################################################################
#--------------------------------------------------- Defining Particles ---------------------------------------------------#

def pdg_id(part_name):
    mapping = {
        "d" : 1,   # down quark
        "u" : 2,   # up quark
        "s" : 3,   # strange quark
        "c" : 4,   # charm quark
        "b" : 5,   # bottom quark
        "t" : 6,   # top quark
        "db" : -1,   # down quark
        "ub" : -2,   # up quark
        "sb" : -3,   # strange quark
        "cb" : -4,   # charm quark
        "bb" : -5,   # bottom quark
        "tb" : -6,   # top quark
        "g" : 21,  # gluon
        "A" : 22,   # photon
        # Add more mappings as needed
    }
    return mapping.get(part_name, 999)  # Default to 999 if pdg_id is not in the mapping

#--------------------------------------------------- Defining PDFs ---------------------------------------------------#

#pdf_set_name = "MSHT20nnlo_as118"

pdf_set_name = "PDF4LHC21_40"


#--------------------------------------------------- Physical Constants ---------------------------------------------------#

GeV = 1

Q2min = 10 * GeV# minimum value for Q^2

GeVtopb = 0.3894e9
GeVtofb = 0.3894e12



E_mu    = 5000      * GeV
m_mu    = 0.10566   * GeV
mp      = 0.938272  * GeV
mW      = 80.369    * GeV
η       =  np.log(E_mu/m_mu * (1+ np.sqrt(1- (m_mu/E_mu)**2)))            # muon rapidity
β       = np.sqrt(1- (m_mu/E_mu)**2)                                       # muon velocity

GF      = 1.1663787e-5 * GeV**(-2)
Gfsqpi = GeVtopb * GF**2 / (np.pi)



pb_to_cm2 = 1e-36
N_Avogadro = 6.022e23

N_neutrino_per_year = 9*1e16

#--------------------------------------------------- Theory Cut ---------------------------------------------------#


W2_min = 4 * GeV**2

Q2_min = 2 * GeV**2


#------------------------------------- Config ----------------------------------#


iterations  = 10
evaluations = 1e3

'''
Binning (logspaced in x and Q2, linearspaced in E)
'''

x_min_bin    = 1e-3     # Minimum x for binning
x_max_bin    = 1        # Maximum x for binning

Q2_min_bin    = 1e1     # Minimum Q2 for binning
Q2_max_bin    = 1e4     # Maximum Q2 for binning

Emax = (1+β)*E_mu/2

E_min_bin    = 0        # Minimum energy for binning
E_max_bin    = Emax      # Minimum energy for binning

#-------------------------------------------------------------------#

########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#------------------------------------ Defining PDFs ---------------------------------#

pset = lhapdf.getPDFSet(pdf_set_name)
pdfs = pset.mkPDFs()


p_central = pdfs[0]

#------------------------------- PDF --------------------------------#
pdf = p_central

#-------------------------------- Neutrino Spectrum ---------------------------------#
#Gamma_mu = GF**2 * m_mu**5 / (192 * np.pi**3)

def dGamma_dE_nu(E_nu):    
    
    
    E_max = (1+β)*E_mu/2
    
    dGamma_normalised = 1 / ( 3 * E_max  ) * (5 - 9 * (  E_nu / E_max ) **2 + 4 * (  E_nu / E_max )**3 ) 
    
    dGamma_normalised = dGamma_normalised * np.where((E_nu > 0) & (E_nu < E_max), 1, 0) # stepfunction for E_nu > E_mu

    return dGamma_normalised 

def dGamma_dE_nubar(E_nu_bar):
    
    E_max = (1+β)*E_mu/2

    dGamma_normalised = 2 / ( E_max  ) * (1 - 3 * (  E_nu_bar / E_max ) **2 + 2 * (  E_nu_bar / E_max )**3 ) 

    dGamma_normalised = dGamma_normalised * np.where((E_nu_bar > 0) & (E_nu_bar < E_max), 1, 0) # stepfunction for E_nu > E_mu

    return dGamma_normalised 

#-------------------------------------
sWSQ = 0.23129  # +0.00004 :  weak mixing angle from PDG 2024 avarage

gZuL = 1/2 - (2/3) * sWSQ
gZuR = - (2/3) * sWSQ

gZdL = -1/2 + (1/3) * sWSQ
gZdR = (1/3) * sWSQ

gZuLSQ = gZuL**2
gZuRSQ = gZuR**2

gZdLSQ = gZdL**2
gZdRSQ = gZdR**2
#------------------------------#

# nu u > nu u
def dsigmah_dxdQ2_NC_nu_u(variables):    
    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2

    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("u"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("u"), x, Q2)-pdfs[0].xfxQ2(pdg_id("u"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZuLSQ + gZuRSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nu(Enu) * sigma

# nu ub > nu ub
def dsigmah_dxdQ2_NC_nu_ub(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("ub"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("ub"), x, Q2)-pdfs[0].xfxQ2(pdg_id("ub"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZuRSQ + gZuLSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nu(Enu) * sigma 


#-------------------------------------
# nu d > nu d
def dsigmah_dxdQ2_NC_nu_d(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2

    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2
    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("d"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("d"), x, Q2)-pdfs[0].xfxQ2(pdg_id("d"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdLSQ + gZdRSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nu(Enu) * sigma

# nu db > nu db
def dsigmah_dxdQ2_NC_nu_db(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2

    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2
    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("db"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("db"), x, Q2)-pdfs[0].xfxQ2(pdg_id("db"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdRSQ + gZdLSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nu(Enu) * sigma


#-------------------------------------
# nu s > nu s
def dsigmah_dxdQ2_NC_nu_s(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2

    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2
    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("s"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("s"), x, Q2)-pdfs[0].xfxQ2(pdg_id("s"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdLSQ + gZdRSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nu(Enu) * sigma

# nu sb > nu sb
def dsigmah_dxdQ2_NC_nu_sb(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2

    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2
    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("sb"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("sb"), x, Q2)-pdfs[0].xfxQ2(pdg_id("sb"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdRSQ + gZdLSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nu(Enu) * sigma


#-------------------------------------
# nu c > nu c
def dsigmah_dxdQ2_NC_nu_c(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("c"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("c"), x, Q2)-pdfs[0].xfxQ2(pdg_id("c"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZuLSQ + gZuRSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nu(Enu) * sigma

# nu cb > nu cb
def dsigmah_dxdQ2_NC_nu_cb(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("cb"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("cb"), x, Q2)-pdfs[0].xfxQ2(pdg_id("cb"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZuRSQ + gZuLSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nu(Enu) * sigma


#-------------------------------------
# nu b > nu b
def dsigmah_dxdQ2_NC_nu_b(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("b"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("b"), x, Q2)-pdfs[0].xfxQ2(pdg_id("b"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdLSQ + gZdRSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nu(Enu) * sigma

# nu bb > nu bb
def dsigmah_dxdQ2_NC_nu_bb(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("bb"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("bb"), x, Q2)-pdfs[0].xfxQ2(pdg_id("bb"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdRSQ + gZdLSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nu(Enu) * sigma



# #-------------------------------------
# nub u > nub u
def dsigmah_dxdQ2_NC_nub_u(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("u"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("u"), x, Q2)-pdfs[0].xfxQ2(pdg_id("u"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZuRSQ + gZuLSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nubar(Enu) * sigma

# nub ub > nub ub
def dsigmah_dxdQ2_NC_nub_ub(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("ub"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("ub"), x, Q2)-pdfs[0].xfxQ2(pdg_id("ub"), x, Q2)
        
    sigma = Gfsqpi * (pdf* (gZuLSQ + gZuRSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nubar(Enu) * sigma


#-------------------------------------
# nub d > nub d
def dsigmah_dxdQ2_NC_nub_d(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("d"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("d"), x, Q2)-pdfs[0].xfxQ2(pdg_id("d"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdRSQ + gZdLSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nubar(Enu) * sigma

# nub db > nub db
def dsigmah_dxdQ2_NC_nub_db(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("db"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("db"), x, Q2)-pdfs[0].xfxQ2(pdg_id("db"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdLSQ + gZdRSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nubar(Enu) * sigma


#-------------------------------------
# nub s > nub s
def dsigmah_dxdQ2_NC_nub_s(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("s"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("s"), x, Q2)-pdfs[0].xfxQ2(pdg_id("s"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdRSQ + gZdLSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nubar(Enu) * sigma

# nub sb > nub sb
def dsigmah_dxdQ2_NC_nub_sb(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("sb"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("sb"), x, Q2)-pdfs[0].xfxQ2(pdg_id("sb"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdLSQ + gZdRSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nubar(Enu) * sigma


#-------------------------------------
# nub c > nub c
def dsigmah_dxdQ2_NC_nub_c(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("c"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("c"), x, Q2)-pdfs[0].xfxQ2(pdg_id("c"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZuRSQ + gZuLSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nubar(Enu) * sigma

# nub cb > nub cb
def dsigmah_dxdQ2_NC_nub_cb(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("cb"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("cb"), x, Q2)-pdfs[0].xfxQ2(pdg_id("cb"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZuLSQ + gZuRSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nubar(Enu) * sigma


#-------------------------------------
# nub b > nub b
def dsigmah_dxdQ2_NC_nub_b(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("b"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("b"), x, Q2)-pdfs[0].xfxQ2(pdg_id("b"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdRSQ + gZdLSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nubar(Enu) * sigma

# nub bb > nub bb
def dsigmah_dxdQ2_NC_nub_bb(variables):    

    x, Q2, Enu = variables
    
    s = 2 * Enu * mp + mp**2
    
    OneMinySQ = (1 - Q2 / (x * s))**2 # this is the factor (1-y)^2

    if replica == 0:
        pdf = pdfs[replica].xfxQ2(pdg_id("bb"), x, Q2)
    else :
        pdf = pdfs[replica].xfxQ2(pdg_id("bb"), x, Q2)-pdfs[0].xfxQ2(pdg_id("bb"), x, Q2)
        
    sigma = Gfsqpi * (pdf * (gZdLSQ + gZdRSQ * OneMinySQ ) ) * np.heaviside(s*x-Q2,0) * np.heaviside(Q2-Q2min,0) / x 

    return dGamma_dE_nubar(Enu) * sigma



#---------------------------------- Cross Section Dictionary -------------------------------#
crossSection = {
        #--------------------------------#
        #----------- nu -----------------#
        'nu_u'   : dsigmah_dxdQ2_NC_nu_u,
        'nu_ub'  : dsigmah_dxdQ2_NC_nu_ub,
        'nu_d'   : dsigmah_dxdQ2_NC_nu_d,
        'nu_db'  : dsigmah_dxdQ2_NC_nu_db,
        'nu_s'   : dsigmah_dxdQ2_NC_nu_s,
        'nu_sb'  : dsigmah_dxdQ2_NC_nu_sb,
        'nu_c'   : dsigmah_dxdQ2_NC_nu_c,
        'nu_cb'  : dsigmah_dxdQ2_NC_nu_cb,
        'nu_b'   : dsigmah_dxdQ2_NC_nu_b,
        'nu_bb'  : dsigmah_dxdQ2_NC_nu_bb,
        #-------------------------------#
        #----------- nubar -------------#
        'nub_u'   : dsigmah_dxdQ2_NC_nub_u,
        'nub_ub'  : dsigmah_dxdQ2_NC_nub_ub,
        'nub_d'   : dsigmah_dxdQ2_NC_nub_d,
        'nub_db'  : dsigmah_dxdQ2_NC_nub_db,
        'nub_s'   : dsigmah_dxdQ2_NC_nub_s,
        'nub_sb'  : dsigmah_dxdQ2_NC_nub_sb,
        'nub_c'   : dsigmah_dxdQ2_NC_nub_c,
        'nub_cb'  : dsigmah_dxdQ2_NC_nub_cb,
        'nub_b'   : dsigmah_dxdQ2_NC_nub_b,
        'nub_bb'  : dsigmah_dxdQ2_NC_nub_bb,
    }

#----------------------------------- Binning ------------------------------------#
x_bins  = np.logspace(np.log10(x_min_bin), np.log10(x_max_bin), num_bins_x + 1)      # Logarithmic bins for x
Q2_bins = np.logspace(np.log10(Q2_min_bin), np.log10(Q2_max_bin), num_bins_Q2 + 1)  # Logarithmic bins for y
E_bins  = np.linspace(E_min_bin, E_max_bin, num_bins_E + 1)                          # Linear bins for E


#----------------------------------- Integration ------------------------------------#

sigma_bin = np.zeros((num_bins_x, num_bins_Q2, num_bins_E))
sigma_bin_error = np.zeros((num_bins_x, num_bins_Q2, num_bins_E))

def compute_bin(m):
    i, j, k = m  
    integral = vegas.Integrator([[x_bins[i], x_bins[i + 1]], [Q2_bins[j], Q2_bins[j + 1]], [E_bins[k], E_bins[k + 1]]])
    
    sigma = integral(crossSection[process], nitn=iterations, neval=evaluations, adapt = True,adapt_to_errors=False)
    # print(f"Bin {i}, {j}, {k}: integral = {sigma.mean} ± {sigma.sdev}")
    return (i, j, k, 
            sigma.mean, sigma.sdev,)



bin_indices = [(i, j, k) for i in range(num_bins_x) for j in range(num_bins_Q2) for k in range(num_bins_E)]


results = [compute_bin(bin) for bin in bin_indices]

# Stampo i risultati
for result in results:
    i, j, k, xsec, xsec_dev = result
    print(f"{i}\t{j}\t{k}\t{xsec}")
    
    


















