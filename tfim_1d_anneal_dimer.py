from tfim_module import TFIM_1d
from noise_model import fourier_noise
from TFIM_1d_dyn import TFIM_general,get_C_fermion,get_C_spin
import numpy as np
import scipy.optimize as op
import os,sys


L = int(sys.argv[1])
T = float(sys.argv[2])
error = float(sys.argv[3])
Nc = int(sys.argv[4])
N_anneal = int(sys.argv[5])
path = sys.argv[6]

J0 = -2*np.ones(L) # contains bonds 
J[::2] = J[1::2]/2.0
J0[-1] = 0.0
h0 = -np.ones(L)

# setting up main object for q-annealing
tfim = TFIM_1d(L)
A = lambda x:(1-x)**2
B = lambda x:x**2
s = lambda t:t/T
tfim_H = TFIM_general(L,A,B,s,(),J=J0,h=h0)

def J_ramp(t,T):
	return (t/T)**2

def h_ramp(t,T):
	return np.exp(-t/(0.3*T))*(1-t/T)**2


def get_samples(Nsamples,omega_min=2.5e-7,omega_max=1,alpha=-0.75):
	def C(omega,omega_min,omega_max,alpha):
		return (omega_min**(1+alpha)-omega**(1+alpha))/(omega_min**(1+alpha)-omega_max**(1+alpha))

	p_list = np.random.uniform(0,1,size=Nsamples)
	omega_list = [] #always include static part
	for p in p_list:
		f = lambda x:C(x,omega_min,omega_max,alpha)-p
		omega_list.append(op.brentq(f,omega_min,omega_max))

	return np.asarray(omega_list)

def get_noise_fourier(Nc,domega,T):
	omega = get_samples(Nc,omega_max=10)
	J = J0+np.random.uniform(0,error,size=J0.shape)
	J_func = fourier_noise(J,J_ramp,error,omega,ramp_args=(T,))
	h_func = lambda t:h0*h_ramp(t,T)
	return J_func,h_func



filename = os.path.join(path,"anneal_noise_L_{}_T_{}_A_{}_Nc_{}.dat".format(L,T,error,Nc))
E0 = J0.sum()
E,V = tfim_H.eigh(h=1.0,J=0.0)
psi_i = V[:,:L].astype(np.complex128)

with open(filename,"a") as IO:
	for i in range(N_anneal):
		print i+1
		J_func,h_func = get_noise_fourier(Nc,error,T)
		psi_f = tfim.anneal(psi_i,0,T,J_func,h_func,atol=1.1e-15,rtol=1.1e-10,solver_name='dop853')

		AA,BB,AB,BA = get_C_fermion(L,psi_f)
		m2 = get_C_spin(AA,BB,AB,BA).real.sum()/L**2
		q = tfim_H.expt_value(psi_f,J=1.0,h=0.0).real - E0

		IO.write("{:30.15e} {:30.15e}\n".format(q,m2))
		IO.flush()
