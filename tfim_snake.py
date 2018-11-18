from tfim_module import TFIM_2d
from noise_model import fourier_noise
import numpy as np
import scipy.optimize as op
import os,sys
from quspin.operators import hamiltonian


L = int(sys.argv[1])
T = float(sys.argv[2])
error = float(sys.argv[3])
Nc = int(sys.argv[4])
N_anneal = int(sys.argv[5])
path = sys.argv[6]

N = 16
Nb = 24

if L not in [4,8,16]:
	raise ValueError

J0 = np.zeros(Nb+N) # contains both bonds and local z fields
J0[N:] = 0.0 # field is set to 0, disorder is added in function
h0 = -np.ones(N)

# setting up main object for q-annealing
tfim = TFIM_2d(4,4)

k = 0
for i,j in tfim.left_iter:
	J0[k] = -1.0
	k += 1

if L==8 or L==16:
	kk = k
	for i,j in tfim.up_iter:
		if i==3 or i==11:
			J0[kk] = -1.0

		kk += 1

if L==16:
	kk = k
	for i,j in tfim.up_iter:
		if i==4:
			J0[kk] = -1.0

		kk += 1



def J_ramp(t,T):
	return (t/T)**2

def h_ramp(t,T):
	return np.exp(-t/(0.3*T))*(1-t/T)**2

def get_samples(Nsamples,omega_min=2.5e-7,omega_max=1,alpha=-0.325):
	def C(omega,omega_min,omega_max,alpha):
		return (omega_min**(1+alpha)-omega**(1+alpha))/(omega_min**(1+alpha)-omega_max**(1+alpha))

	p_list = np.random.uniform(0,1,size=Nsamples)
	omega_list = [] #always include static part
	for p in p_list:
		f = lambda x:C(x,omega_min,omega_max,alpha/2.0)-p
		omega_list.append(op.brentq(f,omega_min,omega_max))

	return np.asarray(omega_list)

def get_noise_fourier(Nc,domega,T):
	omega = get_samples(Nc,omega_max=10)
	J = J0+np.random.normal(0,error,size=J0.shape)
	J_func = fourier_noise(J,J_ramp,error,omega,ramp_args=(T,))
	h_func = lambda t:h0*h_ramp(t,T)
	return J_func,h_func



filename = os.path.join(path,"anneal_noise_noint_L_{}_T_{}_A_{}_Nc_{}.dat".format(L,T,error,Nc))


J_list = []
k = 0
for i,j in tfim.bond_iter:
	if J0[k] != 0:
		J_list.append([J0[k],i,j])
	k += 1

if L == 4:
	M_list =  [[1.0/(4*L**2),i,j] for i in range(L) for j in range(L)]
	M_list += [[1.0/(4*L**2),i+4,j+4] for i in range(L) for j in range(L)]
	M_list += [[1.0/(4*L**2),i+8,j+8] for i in range(L) for j in range(L)]
	M_list += [[1.0/(4*L**2),i+12,j+12] for i in range(L) for j in range(L)]
elif L == 8:
	M_list =  [[1.0/(2*L**2),i,j] for i in range(L) for j in range(L)]
	M_list += [[1.0/(2*L**2),i+8,j+8] for i in range(L) for j in range(L)]
elif L == 16:
	M_list = [[1.0/L**2,i,j] for i in range(L) for j in range(L)]



H_ising = hamiltonian([["zz",J_list]],[],N=N,pauli=True,dtype=np.float64)
M2 = hamiltonian([["zz",M_list]],[],basis=H_ising.basis,dtype=np.float64)


psi_i = np.ones(2**N)/np.sqrt(2**N)
with open(filename,"a") as IO:
	for i in range(N_anneal):
		print i
		J_func,h_func = get_noise_fourier(Nc,error,T)
		psi_f = tfim.anneal(psi_i,0,T,J_func,h_func,atol=1.1e-15,rtol=1.1e-10,solver_name='dop853')

		m2 = M2.expt_value(psi_f).real
		q = H_ising.expt_value(psi_f).real/(16//L) + (L-1)
		IO.write("{:30.15e} {:30.15e}\n".format(q,m2))
		IO.flush()

