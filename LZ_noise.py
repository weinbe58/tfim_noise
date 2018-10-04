from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian,quantum_operator
from tfim_module import fourier_noise,osc_noise
import numpy as np
import scipy.stats as stats
import cProfile
import matplotlib.pyplot as plt

basis = spin_basis_1d(1)
ops_dict = dict(J=[["z",[[1.0,0]]]],h=[["x",[[1.0,0]]]])

O = quantum_operator(ops_dict,basis=basis,dtype=np.float64)




def J_ramp(t,T):
	return -(t/T)**2

def h_ramp(t,T):
	return -(1-t/T)**2

def get_noise_fourier(Nc,domega):
	omega = stats.cauchy.rvs(scale=domega,size=Nc)
	J_func = fourier_noise(J0,J_ramp,error,omega,ramp_args=(T,))
	h_func = fourier_noise(h0,h_ramp,error,omega,ramp_args=(T,))

	return J_func,h_func

def get_noise_osc():
	s = 1.0
	q1 = np.asarray(stats.norm.rvs(scale=0.1,size=J0.shape)+1j*stats.norm.rvs(scale=1.0,size=J0.shape))
	q2 = np.asarray(stats.norm.rvs(scale=0.1,size=h0.shape)+1j*stats.norm.rvs(scale=1.0,size=h0.shape))

	J_func = osc_noise(J0,J_ramp,error,q1,T,s=s,ramp_args=(T,))
	h_func = osc_noise(h0,h_ramp,error,q2,T,s=s,ramp_args=(T,))

	return J_func,h_func


J0 = np.array(0.1)
h0 = np.array(1.0)
T = 100.0
error = 0.01
N_anneal = 100

# J_func,h_func = get_noise_osc()
J_func,h_func = get_noise_fourier(2,0.01)

# times = np.linspace(0,T,10000)
# noise = np.array([[J_func(t),h_func(t)] for t in times])
# print np.squeeze(noise).shape
# plt.plot(times,np.squeeze(noise))
# plt.show()


for i in range(N_anneal):
	print i
	J_func,h_func = get_noise_fourier(1000,0.01)
	# J_func,h_func = get_noise_osc()
	pars = dict(J=(J_func,()),h=(h_ramp,(T,)))

	H = O.tohamiltonian(pars=pars)

	E,V = H.eigh(time=0)
	psi_0 = V[:,0]

	psi_f = H.evolve(psi_0,0,T)


	print np.abs(psi_f)**2




