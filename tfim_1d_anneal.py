from tfim_module import TFIM_1d
from noise_model import fourier_noise
from TFIM_1d_dyn import TFIM_general,get_C_fermion,get_C_spin
import numpy as np
import scipy.optimize as op
import cProfile
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian


L = 100
T = 1000.0
error = 0.01
N_anneal = 10
Nc = 100

Nb = L

J0 = -np.ones(Nb) # contains both bonds and local z fields
J0[-1] = 0.0
h0 = -2*np.ones(L)

# setting up main object for q-annealing
tfim = TFIM_1d(L)
A = lambda x:1-x
B = lambda x:x
tfim_H = TFIM_general(L,A,B,B,(),J=J0)


def J_ramp(t,T):
	return (t/T)**2

def h_ramp(t,T):
	return (1-t/T)**2


def get_samples(Nsamples,omega_min=2.5e-7,omega_max=1,alpha=-0.75):
	def C(omega,omega_min,omega_max,alpha):
		return (omega_min**(1+alpha)-omega**(1+alpha))/(omega_min**(1+alpha)-omega_max**(1+alpha))

	p_list = np.random.uniform(0,1,size=Nsamples)
	omega_list = []
	for p in p_list:
		f = lambda x:C(x,omega_min,omega_max,alpha)-p
		omega_list.append(op.brentq(f,omega_min,omega_max))

	return np.asarray(omega_list)

def get_noise_fourier(Nc,domega,T):
	omega = get_samples(Nc,omega_max=10)
	J_func = fourier_noise(J0,J_ramp,error,omega,ramp_args=(T,))
	h_func = fourier_noise(h0,h_ramp,0.0,np.zeros(1),ramp_args=(T,))
	return J_func,h_func


E,V = tfim_H.eigh(h=1.0,J=0.0)
psi_i = V[:,:L]





J_func,h_func = get_noise_fourier(Nc,error,T)
times = np.linspace(0,T,10001)
J = np.array([J_func(t) for t in times])
plt.plot(times,J)
plt.show()
# exit()


np.random.seed(129391)
J_func,h_func = get_noise_fourier(Nc,error,1)
psi_f = tfim.anneal(psi_i,0,1,J_func,h_func,atol=1.1e-15,rtol=1.1e-10,solver_name='dop853')



J_func,h_func = get_noise_fourier(Nc,error,T)
pr = cProfile.Profile()
pr.enable()
psi_f = tfim.anneal(psi_i,0,T,J_func,h_func,atol=1.1e-15,rtol=1.1e-10,solver_name='dop853')
pr.disable()
pr.print_stats(sort='time')



