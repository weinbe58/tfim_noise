from quspin.basis import spin_basis_1d,tensor_basis,boson_basis_1d
from quspin.operators import hamiltonian,quantum_operator
from quspin.tools.evolution import evolve
from quspin.tools.misc import csr_matvec
from noise_model import fourier_noise
import numpy as np
import cProfile,os,sys,time
import matplotlib.pyplot as plt

def get_samples(Nsamples,omega_min=2.5e-7,omega_max=1,alpha=-0.75):
	import scipy.optimize as op
	def C(omega,omega_min,omega_max,alpha):
		return (omega_min**(1+alpha)-omega**(1+alpha))/(omega_min**(1+alpha)-omega_max**(1+alpha))

	p_list = np.random.uniform(0,1,size=Nsamples)
	omega_list = [] #always include static part
	for p in p_list:
		f = lambda x:C(x,omega_min,omega_max,alpha)-p
		omega_list.append(op.brentq(f,omega_min,omega_max))

	return np.asarray(omega_list)

def SO_MF(t,yin,yout,H0,Hz,eta2,L,T,sigma):
	yin = yin.reshape(yout.shape)
	# get exptation value for local field
	Ly = yin.shape[0]
	hz = []

	for i in range(Ly):
		csr_matvec(Hz,yin[i],out=yout[i],overwrite_out=True)
		mz = (sigma/L)*np.vdot(yin[i],yout[i]).real
		mz += eta2[i](t)
		hz.append(mz*(t/T)**2)
	
	# act with hamiltonian
	for i in range(Ly):
		H0[i]._hamiltonian__omp_SO(t,yin[i],yout[i])

	# act with mean field hamiltonian
	for i in range(Ly):
		hz_mf = 1j*(hz[(i+1)%Ly]+hz[(i-1)%Ly])/2.0
		csr_matvec(Hz,yin[i],a=hz_mf,out=yout[i],overwrite_out=False)

	return yout.ravel()


def anneal_MF(L,T,sigma=0.01,path=".",n_chains=1):
	ti = time.time()

	print "creating basis"
	basis = spin_basis_1d(L,pauli=True,kblock=0,pblock=1)

	hx_list = [[-1,i] for i in range(L)]
	hz_list = [[1,i] for i in range(L)]
	J_list = [[-1,i,(i+1)%L] for i in range(L)]
	M_list = [[1.0/L**2,i,j] for i in range(L) for j in range(L)]

	ops_dict = dict(J=[["zz",J_list]],h=[["x",hx_list]])
	H0 = quantum_operator(ops_dict,basis=basis,dtype=np.float64)
	Hz = hamiltonian([["z",hz_list]],[],basis=basis,dtype=np.float64)
	M2 = hamiltonian([["zz",M_list]],[],basis=basis,dtype=np.float64)

	def B(t,T):
		return (1-t/T)**2

	def A(t,T,J0,eta):
		return (J0+eta(t))*(t/T)**2

	_,psi0 = H0.eigsh(k=1,which="SA",pars=dict(J=0,h=1))

	psi0 = psi0.astype(np.complex128).ravel()

	y0 = np.hstack([psi0 for i in range(n_chains)])
	yout = y0.reshape((n_chains,basis.Ns)).copy()

	for i in range(3):
		H_list = []
		eta2_list = []
		for j in range(n_chains):
			omega1 = get_samples(1000,omega_max=10)
			omega2 = get_samples(1000,omega_max=10)
			eta1 = fourier_noise(0.0,lambda t:1.0,sigma,omega1)
			eta2 = fourier_noise(0.0,lambda t:1.0,sigma,omega2)

			# if j==0:
			# 	pars = dict(J=(A,(T,1.0,eta1)),h=(B,(T,)))
			# else:
			# 	pars = dict(J=(A,(T,sigma,eta1)),h=(B,(T,)))

			pars = dict(J=(A,(T,1.0,eta1)),h=(B,(T,)))

			H_list.append(H0.tohamiltonian(pars=pars))
			eta2_list.append(eta2)



		f_params = (yout,H_list,Hz.tocsr(),eta2_list,L,T,sigma)

		psif = evolve(y0.reshape((-1,basis.Ns)),0,T,SO_MF,f_params=f_params,solver_name="dop853",
			atol=1.1e-15,rtol=1.1e-15)

		psif = psif[0]

		q,m2 = H0.matrix_ele(psif,psif,pars=dict(J=1,h=0),diagonal=True).real+L,M2.expt_value(psif).real

		line = "{:30.15e} {:30.15e}".format(q,m2)

		print i+1,line



L = 4
anneal_MF(L,0.5*L**2,0.01,n_chains=3)

L = 6
anneal_MF(L,0.5*L**2,0.01,n_chains=3)

L = 8
anneal_MF(L,0.5*L**2,0.01,n_chains=3)

L = 10
anneal_MF(L,0.5*L**2,0.01,n_chains=3)
