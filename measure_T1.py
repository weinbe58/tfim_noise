from tfim_module import TFIM_2d
from noise_model import fourier_noise
import numpy as np
import scipy.optimize as op
import os,sys
from quspin.operators import hamiltonian
import matplotlib.pyplot as plt


Lx = 10
s = 0.5
error = 0.05
Nc = 1000
N_anneal = 20


N = Lx
Nb = (Lx-1)


J0 = -np.ones(Nb+N) # contains both bonds and local z fields
J0[N:] = 0.0 # field is set to 0, disorder is added in function
error_J = error*np.ones_like(J0)
error_J[N:] = 0.0
h0 = -np.ones(N)
error_h = error*np.ones_like(h0)

# setting up main object for q-annealing
tfim = TFIM_2d(Lx,1)


def J_ramp(t,s):
	return s**2

def h_ramp(t,s):
	return np.exp(-s/(0.3))*(1-s)**2

def get_samples(Nsamples,omega_min=2.5e-7,omega_max=1,alpha=-0.75):
	def C(omega,omega_min,omega_max,alpha):
		return (omega_min**(1+alpha)-omega**(1+alpha))/(omega_min**(1+alpha)-omega_max**(1+alpha))

	p_list = np.random.uniform(0,1,size=Nsamples)
	omega_list = [] #always include static part
	for p in p_list:
		f = lambda x:C(x,omega_min,omega_max,alpha)-p
		omega_list.append(op.brentq(f,omega_min,omega_max))

	return np.asarray(omega_list)

def get_noise_fourier(Nc,domega,s):
	omega = get_samples(Nc,omega_max=10,alpha=1.0)
	J = J0+np.random.normal(0,error,size=J0.shape)
	J_func = fourier_noise(J,J_ramp,error_J,omega,ramp_args=(s,))
	h_func = fourier_noise(h0,h_ramp,error_h,omega,ramp_args=(s,))
	# h_func = lambda t:h0*h_ramp(t,s)
	return J_func,h_func

def bootstrap_mean(Os,n_bs=100):
	Os = np.asarray(Os)
	avg = np.nanmean(Os,axis=0)
	n_real = Os.shape[0]

	bs_iter = (np.nanmean(Os[np.random.randint(n_real,size=n_real)],axis=0) for i in range(n_bs))
	diff_iter = ((bs-avg)**2 for bs in bs_iter)
	err = np.sqrt(sum(diff_iter)/n_bs)

	return avg,err


Sz = hamiltonian([["z",[[1.0,Lx//2]]]],[],N=N,dtype=np.float32,pauli=True)

times = np.linspace(0,100,100)

Cz_list = []

for i in range(N_anneal):
	print i+1
	psi_i = np.random.normal(0,1,size=2**N)
	psi_i /= np.linalg.norm(psi_i)
	psi_z = Sz.dot(psi_i)

	J_func,h_func = get_noise_fourier(Nc,error,s)


	psi_1 = tfim.anneal(psi_i,0,times,J_func,h_func,atol=1.1e-15,rtol=1.1e-10,solver_name='dop853')
	psi_2 = tfim.anneal(psi_z,0,times,J_func,h_func,atol=1.1e-15,rtol=1.1e-10,solver_name='dop853')

	Cz = Sz.matrix_ele(psi_2,psi_1,diagonal=True).real
	Cz_list.append(Cz)




Cz_list = np.asarray(Cz_list)

Cz,dCz = bootstrap_mean(Cz_list)

plt.errorbar(times,Cz,dCz,marker=".")
plt.show()

