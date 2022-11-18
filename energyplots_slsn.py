import redback 
import pandas as pd
from bilby.core.prior import Uniform, Gaussian, PriorDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from redback.constants import day_to_s
from redback.model_library import all_models_dict

smdat=np.loadtxt('lc_L46large_S21.txt')

model = 'slsn_bolometric'
function = all_models_dict[model]
time_sim = np.logspace(2, 8, 1000)/day_to_s

mej=10.0
kappa=0.1
kappa_gamma=0.1
#tf=6000
#output_format='flux_density'
#frequency=redback.utils.bands_to_frequency('g')[0]

#L46_small

P_1e52=1.6
P_3e51=2.9
P_1e51=5.1
B_L46s=1.8*np.sqrt(10)
B_L46m=1.8
B_L46l=1.8/np.sqrt(10)
B_L48s=18.3*np.sqrt(10)
B_L48m=18.3
B_L48l=18.3/np.sqrt(10)

theta_pb=1.57
mass_ns=1.4
v_ej=5e3

l46_s=function(time_sim, p0=P_1e51, bp=B_L46s, mass_ns=mass_ns, theta_pb=theta_pb, mej=mej, vej=v_ej, kappa=kappa, kappa_gamma=kappa_gamma)
l46_m=function(time_sim, p0=P_3e51, bp=B_L46m, mass_ns=mass_ns, theta_pb=theta_pb, mej=mej, vej=v_ej, kappa=kappa, kappa_gamma=kappa_gamma)
l46_l=function(time_sim, p0=P_1e52, bp=B_L46l, mass_ns=mass_ns, theta_pb=theta_pb, mej=mej, vej=v_ej, kappa=kappa, kappa_gamma=kappa_gamma)

l48_s=function(time_sim, p0=P_1e51, bp=B_L48s, mass_ns=mass_ns, theta_pb=theta_pb, mej=mej, vej=v_ej, kappa=kappa, kappa_gamma=kappa_gamma)
l48_m=function(time_sim, p0=P_3e51, bp=B_L48m, mass_ns=mass_ns, theta_pb=theta_pb, mej=mej, vej=v_ej, kappa=kappa, kappa_gamma=kappa_gamma)
l48_l=function(time_sim, p0=P_1e52, bp=B_L48l, mass_ns=mass_ns, theta_pb=theta_pb, mej=mej, vej=v_ej, kappa=kappa, kappa_gamma=kappa_gamma)

plt.close('all')
plt.semilogy(time_sim, l46_s,'r-',label='L46s')
plt.semilogy(time_sim, l46_m,'b-',label='L46m')
plt.semilogy(time_sim, l46_l,'g-',label='L46l')
plt.semilogy(time_sim, l48_s,'m-',label='L48s')
plt.semilogy(time_sim, l48_m,'c-',label='L48m')
plt.semilogy(time_sim, l48_l,'y-',label='L48l')
plt.semilogy([1,1],[1,1],'w-',label=' ')
plt.semilogy(smdat[:,0]/day_to_s, smdat[:,1]+smdat[:,12],'k-',label='SM21 0$^{\\rm o}$')
plt.semilogy(smdat[:,0]/day_to_s, smdat[:,6]+smdat[:,17],'k:',label='SM21 90$^{\\rm o}$')
plt.xlim(-5,200)
plt.ylim(1e38,1e46)
plt.xlabel("Time (days)")
plt.ylabel("Bolometric Luminosity (erg s$^{-1}$)")
plt.legend()
plt.savefig('SLSNbolo.pdf')
plt.show()

model = 'basic_magnetar'
function = all_models_dict[model]

magl46_s=function(time_sim*day_to_s, p0=P_1e51, bp=B_L46s, mass_ns=mass_ns, theta_pb=theta_pb)
magl46_m=function(time_sim*day_to_s, p0=P_3e51, bp=B_L46m, mass_ns=mass_ns, theta_pb=theta_pb)
magl46_l=function(time_sim*day_to_s, p0=P_1e52, bp=B_L46l, mass_ns=mass_ns, theta_pb=theta_pb)

magl48_s=function(time_sim*day_to_s, p0=P_1e51, bp=B_L48s, mass_ns=mass_ns, theta_pb=theta_pb)
magl48_m=function(time_sim*day_to_s, p0=P_3e51, bp=B_L48m, mass_ns=mass_ns, theta_pb=theta_pb)
magl48_l=function(time_sim*day_to_s, p0=P_1e52, bp=B_L48l, mass_ns=mass_ns, theta_pb=theta_pb)

plt.close('all')
plt.loglog(time_sim*day_to_s, magl46_s,'r-',label='L46s')
plt.loglog(time_sim*day_to_s, magl46_m,'b-',label='L46m')
plt.loglog(time_sim*day_to_s, magl46_l,'g-',label='L46l')
plt.loglog(time_sim*day_to_s, magl48_s,'m-',label='L48s')
plt.loglog(time_sim*day_to_s, magl48_m,'c-',label='L48m')
plt.loglog(time_sim*day_to_s, magl48_l,'y-',label='L48l')
#plt.xlim(1e,200)
#plt.ylim(1e38,1e46)
plt.xlabel("Time (seconds)")
plt.ylabel("Magnetar Luminosity (erg s$^{-1}$)")
plt.legend()
plt.show()
