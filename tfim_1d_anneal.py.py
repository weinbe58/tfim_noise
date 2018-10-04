from tfim_module import TFIM_1d
from noise_model import fourier_noise,osc_noise
import numpy as np
import scipy.stats as stats
import cProfile
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian


L = 10
T = 1000.0
error = 0.02*6
N_anneal = 10
Nc = 1000

Nb = L

J0 = -6*np.ones(Nb) # contains both bonds and local z fields
J0[L] = -error # fields average coupling is 0.0
h0 = -12*np.ones(N)

# setting up main object for q-annealing
tfim = TFIM_1d(L)


def J_ramp(t,T):
	return (t/T)**2

def h_ramp(t,T):
	return (1-t/T)**2

def get_noise_fourier(Nc,domega):
	omega = stats.cauchy.rvs(scale=domega,size=Nc)
	J_func = fourier_noise(J0,J_ramp,error,omega,ramp_args=(T,))
	h_func = fourier_noise(h0,h_ramp,0.0,np.zeros(1),ramp_args=(T,))
	return J_func,h_func

def get_noise_osc():
	s = 1.0
	q1 = stats.norm.rvs(scale=0.1,size=J0.shape)+1j*stats.norm.rvs(scale=1.0,size=J0.shape)
	q2 = stats.norm.rvs(scale=0.1,size=h0.shape)+1j*stats.norm.rvs(scale=1.0,size=h0.shape)
	J_func = osc_noise(J0,J_ramp,error,q1,T,s=s,ramp_args=(T,))
	h_func = osc_noise(h0,h_ramp,error,q2,T,s=s,ramp_args=(T,))

	return J_func,h_func







for i in range(N_anneal):
	J_func,h_func = get_noise_fourier(Nc,error)

	psi_f = tfim.anneal(psi_i,0,T,J_func,h_func,atol=1.1e-7,rtol=1.1e-7,solver_name='dop853')

	print psi_f.conj().dot(H_ising*psi_f).real,
	print psi_f.conj().dot(M*M*psi_f).real


