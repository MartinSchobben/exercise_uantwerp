import numpy as np
from matplotlib import pyplot as plt
import phreeqpython as ph
import math
import chemw
import os
from scipy.integrate import odeint

# start session
pp = ph.PhreeqPython(os.path.join(os.getcwd(), 'database', 'T_H.DAT'))

# get molecular weights
chem_mw = chemw.ChemMW(verbose = False, printing = True)

# boundary conditions
mass_basalt = 1000 # g
volume = 1000 # l
pH = 7
Temp = 25 # degree Celcius
pCO2 = 2 # bar
logCO2 = math.log10(pCO2)

# initiate solution 1
solution1 = pp.add_solution({})
# check alkalinity
solution1.total('HCO3-', 'mmol')  + 2 * solution1.total('CO3-', 'mmol')

# CO2 in equilibrium
solution1 = solution1.equalize(['CO2(g)'], [logCO2])
# check alkalinity
solution1.total('HCO3-', 'mmol')  + 2 * solution1.total('CO3-', 'mmol')

# initiate solution 2
solution2 = pp.add_solution({})
# check alkalinity
solution2.total('HCO3-', 'mmol')  + 2 * solution2.total('CO3-', 'mmol')

# CO2 in equilibrium
solution2 = solution2.equalize(['CO2(g)'], [logCO2])
# check alkalinity
solution2.total('HCO3-', 'mmol')  + 2 * solution2.total('CO3-', 'mmol')

# keep gas at fixed pressure
fp = pp.add_gas({'CO2(g)': 0,}, pressure=pCO2, fixed_pressure=True)
solution1.interact(fp.copy())
solution2.interact(fp.copy())

# specific surface area 
ssa1 = 5 # m2 / g
ssa2 = 1 # m2 / g

# mineral fractions basalt1
frac_for1 = 0.05
mass_for1 =  mass_basalt * frac_for1 # g
frac_plag1 = 0.5
mass_plag1 = mass_basalt * frac_plag1 # g
frac_diop1 = 0.45
mass_diop1 = mass_basalt * frac_diop1 # g

# mineral fractions basalt2
frac_for2 = 0.15
mass_for2 =  mass_basalt * frac_for2 # g
frac_plag2 = 0.35
mass_plag2 = mass_basalt * frac_plag2 # g
frac_diop2 = 0.5
mass_diop2 = mass_basalt * frac_diop2 # g

# Olivine (Forsterite = Mg2SiO4 + 4H+ = 2Mg+2 + H4SiO4)
name_for = 'Forsterite'
formula_for = 'Mg2SiO4'
amu_for = chem_mw.mass(formula_for)
dens_for = 3.32 # g mineral per cm3 mineral

# basalt 1
M0_for1 = mass_for1 / float(amu_for) # mol

# basalt 2
M0_for2 = mass_for2 / float(amu_for) # mol

# Plagioclase (Anorthite = CaAl2Si2O8 + 8H2O = Ca+2 + 2Al(OH)4- + 2H4SiO4)
name_plag = 'Anorthite'
formula_plag = 'CaAl2Si2O8' 
amu_plag = chem_mw.mass(formula_plag)
dens_plag = 2.73 # g mineral per cm3 mineral

# basalt 1
M0_plag1 = mass_plag1 / float(amu_plag) # mol 

# basalt 2
M0_plag2 = mass_plag2 / float(amu_plag) # mol 

# Diopside (Augite = CaMgSi2O6 + 4H+ + 2H2O = Ca+2 + Mg+2 + 2H4SiO4)
name_diop = 'Diposide'
formula_diop = 'CaMgSi2O6'
amu_diop = chem_mw.mass(formula_diop)
dens_diop = 3.4 # g mineral per cm3 mineral

# basalt 1
M0_diop1 = mass_diop1 / float(amu_diop) # mol

# basalt 2
M0_diop2 = mass_diop2 / float(amu_diop) # mol

# Mineral specific surface area
def calc_surface(mass, dens, total_ssa):
    """ 
    Parameters
    ------------
    mass = array of mineral masses in grams
    dens = array of mineral densities in gram per cm3 
    total_ssa = specific surface area of total rock (cm2 / g)

    Description
    ---------------
    Use volume fractions to caclculate relative SSA (RSSA) for 
    each of the minerals of the rock.

    SSA_mineral (cm2 / g) = RSSA_mineral * SSA_tot (cm2 / g)
    A_mineral = mass_mineral * SSA_mineral

    Output
    -------
    initial surface of mineral m2
    
    """
    
    # initialise arrays
    vol_frac = np.array([])
    rssa = np.array([])
    
    # calculate volume fraction
    tot_vol_frac = mass * dens # volume fraction
    for i in range(len(mass)):
        vol_frac = np.append(vol_frac, (mass[i] * dens[i]) / tot_vol_frac.sum())

    # calculate relative ssa
    tot_rssa = vol_frac ** (2 / 3)
    for i in range(len(vol_frac)):
        rssa = np.append(rssa, (vol_frac[i]  ** (2 / 3)) / tot_rssa.sum()) 

    return tuple(rssa * total_ssa *  mass)

# arrays of masses 
mass1 = np.array([mass_for1, mass_plag1, mass_diop1])
mass2 = np.array([mass_for2, mass_plag2, mass_diop2])
# arrays of densities
dens = np.array([dens_for, dens_plag, dens_diop])
# mineral specific surface area
A_for1, A_plag1, A_diop1 = calc_surface(mass1, dens, ssa1) # basalt 1
A_for2, A_plag2, A_diop2 = calc_surface(mass2, dens, ssa2) # basalt 2

def calc_rate_const(k_acid, eapp_acid, n_acid, k_neut, eapp_neut, k_base, eapp_base,  n_base):
    def rate_const(sol) :
        """ 
        R = gas constant 8.314e-3 J / K / mol   1.987 cal / mol * K
        eapp activation energy (Kj/mol) 
        T = K
        k = moles/m2/s
        """
        temp = sol.copy()
        dif_temp = 1 / (temp.temperature + 273.15) - 1 / 298.15
        hplus = temp.total("H+", 'mol')

        r_acid = k_acid * math.exp((-eapp_acid / 8.314e-3) * dif_temp) * (hplus ** n_acid)
        r_neut = k_neut * math.exp((-eapp_neut / 8.314e-3) * dif_temp)
        r_base = k_base * math.exp((-eapp_base / 8.314e-3) * dif_temp) * (hplus ** n_base)
    
        return r_acid + r_neut + r_base
    return rate_const
 

# function to calculate rate constant diopside
k_diop = calc_rate_const(k_acid = 10 ** -6.36, eapp_acid = 96.1, n_acid = 0.71,  k_neut = 10 ** -11.11, eapp_neut = 40.6, k_base = 0, eapp_base = 0,  n_base = 0)
k_for = calc_rate_const(k_acid = 10 ** -6.85, eapp_acid = 67.2, n_acid = 0.74,  k_neut = 10 ** -10.64, eapp_neut = 79, k_base = 0, eapp_base = 0,  n_base = 0)
k_plag = calc_rate_const(k_acid = 10 ** -7.87, eapp_acid = 42.1, n_acid = 0.626,  k_neut = 10 ** -10.91, eapp_neut = 45.2, k_base = 0, eapp_base = 0,  n_base = 0)

# kinetic dissolution
def ratefun(dm, time, sol, form, phase, m0, A0, V, kfun):
    """ 
    parameters
    --------
    sol = The solution at timestep t
    dm = difference in moles of for timestep dt
    m0 = Moles of initial phase
    A0 = Initial surface m2 (calculate with mineral specific surface)
    V = Volume in liters
    species = String of phase name,
    kfun = function to calculate rate constant

    Constants
    ---------
    k rate constant = 10-13.7 mol/m2/s (25 C)
    SR is the Saturation Ratio for the phase

    Output
    --------
    Dissolution rate mol/liter/sec
    """
    # save intermediate
    temp = sol.copy()

    # add to solution
    temp.add(form, dm[0], "mol")

    # moles of phase for timestep t
    m = m0 - dm[0]
    
    # rate constant
    k = kfun(temp)

    print("Rate constant: " + str(k))
    
    # rate at timestep t
    if sol.sr(phase) < 0 or m >= 0:
        rate = (A0 / V) * (m / m0) ** 0.67 * k * (1 - sol.sr(phase))
    else:
        rate = 0
    
    temp.forget() # cleanup the no longer needed temporary solution
    return rate

# solve ODE 
def solve_kinetic(sol, time, form, phase, m0, A0, V, kfun):

    # integrate
    yy = odeint(ratefun, 0, time, args=(sol, form, phase, m0, A0, V, kfun))
    dif_yy = np.insert(np.diff(yy[:,0]), 0, 0)

    # impact on alkalinity
    alk = []
    for i in range(len(time)):
        t = tt[i]
        sol.add(form, dif_yy[i])
        alk.append(sol.total('HCO3-', 'mmol')  + 2 * sol.total('CO3-', 'mmol'))

    return np.array(alk), yy[:,0]

# seconds in week
weeks = 7 * 24 * 3600
nmax = 2 # weeks
tt = np.linspace(0, nmax * weeks, 25)

# solve differential equation basalt1
# add forsterite
y_for1, M_for1 = solve_kinetic(solution1, tt, formula_for, name_for, M0_for1, A_for1, volume, k_for)

# add ca plagioclase
y_plag1, M_plag1 = solve_kinetic(solution1, tt, formula_plag, name_plag, M0_plag1, A_plag1, volume, k_plag)

# add diopside
y_diop1, M_diop1 = solve_kinetic(solution1, tt, formula_diop, name_diop, M0_diop1, A_diop1, volume, k_diop)

# solve differential equation basalt2
# add forsterite
y_for2, M_for2 = solve_kinetic(solution2, tt, formula_for, name_for, M0_for2, A_for2, volume, k_for)

# add ca plagioclase
y_plag2, M_plag2 = solve_kinetic(solution2, tt, formula_plag, name_plag, M0_plag2, A_plag2, volume, k_plag)

# add diopside
y_diop2, M_diop2 = solve_kinetic(solution2, tt, formula_diop, name_diop, M0_diop2, A_diop2, volume, k_diop)

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(tt / weeks, y_diop1 + y_plag1 + y_for1, label = 'Alkalinity')
ax2.plot(tt / weeks, y_diop2 + y_plag2 + y_for2, label = 'Alkalinity')

# decoration
ax1.set_xlabel('weeks')
ax2.set_xlabel('weeks')
ax1.set_ylabel('mmol/l')
ax1.set_title('Dissolution basalt 1')
ax2.set_title('Dissolution basalt 2')

# bar plot of difference
fig, ax = plt.subplots()

# difference in alkalinity over two weeks
diff1 = (y_diop1[-1] + y_for1[-1] + y_plag1[-1]) - (y_diop1[0] + y_for1[0] + y_plag1[0])
diff2 = (y_diop2[-1] + y_for2[-1] + y_plag2[-1]) - (y_diop2[0] + y_for2[0] + y_plag2[0])
diffs = [diff1, diff2]
names = ['Basalt 1', 'Basalt2']

# barplot 
ax.bar(names, diffs, label=names, color=['blue', 'red'])
ax.set_ylabel('difference alkalinity (mmol/l) after 2 weeks')
ax.set_title('Dissolution of basalts')

# bar plot of difference
fig, ax = plt.subplots()

# difference in alkalinity over two weeks
diff1 = (M_diop1[-1] + M_for1[-1] + M_plag1[-1]) - (M_diop1[0] + M_for1[0] + M_plag1[0])
diff2 = (M_diop2[-1] + M_for2[-1] + M_plag2[-1]) - (M_diop2[0] + M_for2[0] + M_plag2[0])
diffs = [diff1 * 1e3, diff2 * 1e3]
names = ['Basalt 1', 'Basalt2']

# barplot 
ax.bar(names, diffs, label=names, color=['blue', 'red'])
ax.set_ylabel('difference mmol basalt after 2 weeks')
ax.set_title('Dissolution of basalts')

plt.show()

year = 365 * 24 * 3600

M01 = M0_diop1 + M0_for1 + M0_plag1 # moles start
M02 = M0_diop2 + M0_for2 + M0_plag2 # moles start

out = (np.array([M01, M02]) * 1e3  * 0.99) / (np.array(diffs) / (2 * weeks)) # seconds
out / year