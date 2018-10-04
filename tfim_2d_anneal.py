from tfim_module import TFIM_2d
from noise_model import fourier_noise,osc_noise
import numpy as np
import scipy.stats as stats
import cProfile
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian


Lx = 4
Ly = 1
T = 1000.0
error = 0.02*6
N_anneal = 10
Nc = 1

N = Lx*Ly
Nb = Lx*(Ly-1)+Ly*(Lx-1)

J0 = -6*np.ones(Nb+N) # contains both bonds and local z fields
J0[N:] = -error # fields average coupling is 0.0
h0 = -12*np.ones(N)

# setting up main object for q-annealing
tfim = TFIM_2d(Lx,Ly)


def J_ramp(t,T):
	return (t/T)**2

def h_ramp(t,T):
	return (1-t/T)**2

def get_noise_fourier(Nc,domega):
	omega = stats.cauchy.rvs(scale=domega,size=Nc)
	J_func = fourier_noise(J0,J_ramp,error,omega,ramp_args=(T,))
	h_func = fourier_noise(h0,h_ramp,error,omega,ramp_args=(T,))
	return J_func,h_func

def get_noise_osc():
	s = 1.0
	q1 = stats.norm.rvs(scale=0.1,size=J0.shape)+1j*stats.norm.rvs(scale=1.0,size=J0.shape)
	q2 = stats.norm.rvs(scale=0.1,size=h0.shape)+1j*stats.norm.rvs(scale=1.0,size=h0.shape)
	J_func = osc_noise(J0,J_ramp,error,q1,T,s=s,ramp_args=(T,))
	h_func = osc_noise(h0,h_ramp,error,q2,T,s=s,ramp_args=(T,))

	return J_func,h_func


J_list = [[-1.0,i,j] for i,j in tfim.up_iter]
J_list.extend([[-1.0,i,j] for i,j in tfim.left_iter])

M_list = [[1.0,i] for i in range(N)]

M = hamiltonian([["z",M_list]],[],N=N,pauli=True,dtype=np.float32).diagonal()/N
H_ising = hamiltonian([["zz",J_list]],[],N=N,pauli=True,dtype=np.float32).diagonal()
H_ising -= H_ising.min()
H_ising /= N



# J_func,h_func = get_noise_osc()
J_func,h_func = get_noise_fourier(10,error)

psi_i = np.ones(2**N)/np.sqrt(2**N)

# times = np.linspace(0,T,1000)
# noise = np.array([(J_func(t)[0],h_func(t)[0]) for t in times])
# plt.plot(times,noise)
# plt.show()
# exit()


for i in range(N_anneal):
	J_func,h_func = get_noise_fourier(Nc,error)
	# J_func,h_func = get_noise_osc()

	psi_f = tfim.anneal(psi_i,0,T,J_func,h_func,atol=1.1e-7,rtol=1.1e-7,solver_name='dop853')

	print psi_f.conj().dot(H_ising*psi_f).real,
	print psi_f.conj().dot(M*M*psi_f).real


