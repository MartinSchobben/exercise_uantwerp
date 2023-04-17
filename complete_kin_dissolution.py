""" 
solution1 = pp.add_solution({})

t = np.array([])
y = []

year = 365*24*3600

for time, sol in solution1.kinetics('SiO2', 
                                    rate_function=ratefun, 
                                    time=np.linspace(0,5*year, 15), 
                                    m0=158.5, 
                                    args=(23.13,0.16)):
    t = np.append(t, time)
    y.append(sol.total_element('Si', units='mmol'))

from matplotlib import pyplot as plt


plt.figure(figsize=[10,5])
plt.plot(t/year,y, 'rs-')
plt.xlim([0,5])
plt.ylim([0,0.12])
plt.xlabel('Years')
plt.ylabel('mmol/l')
plt.title('Quartz Dissolution')
plt.grid() """
""" 
def rate_olivine(sol, olivine_dissolved, M0, rssa, r_conc, wporosity, v):
    
    # initialize vector
    rate = 0
    M = M0 - olivine_dissolved
    if M >= 0: #or sol.si("Olivine") < 0:
        a0 = rssa * r_conc * wporosity
        dif_temp = 1 / sol.temperature - 1 / 298.15
        k_acid = 10 ** (-6.85)
        eapp_acid = 67.2
        n_acid = 0.47
        k_neut = 10 ** (-10.64)
        eapp_neut = 79
        k_base = 0
        eapp_base = 0
        n_base = 0
        hplus = sol.total("H+", 'mol')
        r_acid = k_acid * math.exp((-eapp_acid / 8.314e-3) * dif_temp) * (hplus ** n_acid)
        r_neut = k_neut * math.exp((-eapp_neut / 8.314e-3) * dif_temp)
        r_base = k_base * math.exp((-eapp_base / 8.314e-3) * dif_temp) * (hplus ** n_base)
        r_all = r_acid + r_neut + r_base
        rate = (a0 / v) * (M / M0) ** 0.67 * r_all * (1 - sol.sr("Olivine"))
    return rate """