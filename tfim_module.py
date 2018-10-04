from numba import jit,uint64
import numpy as np
from quspin.tools.evolution import evolve
from scipy.integrate import solve_ivp


@jit(nopython=True)
def _2d_ising_term(N,i,j,ising):
	Ns = (1<<N)
	for k in range(Ns):
		ising[k] += 1 - ((((k>>i)^(k>>j))&1)<<1) 

@jit(nopython=True)
def _2d_z_term(N,i,field):
	Ns = (1<<N)
	for k in range(Ns):
		field[k] += (((k>>i)&1)<<1)-1

@jit(nopython=True)
def _2d_x_field(yin,yout,h): # adds to yout.
	N = len(h)
	Ns = (1<<N)

	for i in range(Ns):
		b = uint64(1) # use this number fo flip bit to get column index
		ME = 0
		for j in range(N):
			ME += h[j]*yin[i^b] # x-field action
			b <<= 1 # shift flipping fit to the right

		yout[i] += ME


@jit(nopython=True)
def _2d_diagonal(yin,yout,diag_signs,J):
	N = diag_signs.shape[1]
	Ns = diag_signs.shape[0]
	for i in range(Ns):
		ME = 0
		for j in range(1,N):
			ME += J[j]*diag_signs[i,j]

		yout[i] = ME*yin[i]




def _SE_2d(t,psi_in,psi_out,diag_signs,diag,J_func,h_func):

	J = 1j*J_func(t)
	h = 1j*h_func(t)

	_2d_diagonal(psi_in,psi_out,diag_signs,J)
	_2d_x_field(psi_in,psi_out,h)

	return psi_out # return view of complex array as double for ode solver. 

class TFIM_2d(object):
	def __init__(self,Lx,Ly,open_bc=True):
		N = Lx*Ly
		Ns = 2**N

		self._N = N
		self._Ns = Ns


		self._diag = np.zeros(self._Ns,dtype=np.complex128)
		self._psi_out = self._diag.copy()

		if open_bc:
			self._up_iter = [(i+Lx*j,i+Lx*(j+1)) for i in range(Lx) for j in range(Ly-1)]
			self._left_iter = [(i+Lx*j,(i+1)+Lx*j) for i in range(Lx-1) for j in range(Ly)]
			self._diag_signs = np.zeros((2**N,N+Lx*(Ly-1)+(Lx-1)*Ly),dtype=np.int8)
			
			k = 0
			for i,j in self._left_iter:
				_2d_ising_term(N,i,j,self._diag_signs[:,k])
				k += 1
				

			for i,j in self._up_iter:
				_2d_ising_term(N,i,j,self._diag_signs[:,k])
				k += 1

			for i in range(N):
				_2d_z_term(N,i,self._diag_signs[:,k])
				k += 1

		else:
			self._up_iter = [(i+Lx*j,i+Lx*((j+1)%Ly)) for i in range(Lx) for j in range(Ly)]
			self._left_iter = [(i+Lx*j,(i+1)%Lx+Lx*j) for i in range(Lx) for j in range(Ly)]
			self._diag_signs = np.zeros((2**N,3*N),dtype=np.int8)
			
			k = 0
			for i,j in self._left_iter:
				_2d_ising_term(N,i,j,self._diag_signs[:,k])
				k += 1
				

			for i,j in self._up_iter:
				_2d_ising_term(N,i,j,self._diag_signs[:,k])
				k += 1

			for i in range(N):
				_2d_z_term(N,i,self._diag_signs[:,k])
				k += 1

	@property
	def up_iter(self):
		return iter(self._up_iter)

	@property
	def left_iter(self):
		return iter(self._left_iter)

	def dot(self,psi_in,J,h,psi_out=None):
		if psi_out is None:
			psi_out = psi_in.copy()

		_2d_diagonal(psi_in,psi_out,diag_signs,J)
		_2d_x_field(psi_in,psi_out,h)

		return psi_out

	def anneal(self,psi_i,ti,tf,J_func,h_func,**solver_args):
		f_params = (self._psi_out,self._diag_signs,self._diag,J_func,h_func)
		return evolve(psi_i,ti,tf,_SE_2d,f_params=f_params,**solver_args)





@jit(nopython=True)
def _1d_hamiltonian(yin,yout,J,h):
	L = len(h)
	for i in range(L):
		ip = (i+1)%L
		im = (i-1)%L
		yout[i,:]   =  JJ[im]*(yin[im,:] - yin[L+im,:]) + JJ[i]*(yin[ip,:] + yin[L+ip,:])  - yin[i,:]*h[i]	
		yout[L+i,:] =  yin[L+i,:]*h[i] - (JJ[im]*(yin[L+im,:] - yin[im,:]) + JJ[i]*(yin[L+ip,:] + yin[ip,:])) 

def _SE_1d(t,psi_in,psi_out,J_func,h_func):
	J = -1j*J_func(t)
	h = -1j*h_func(t)
	_1d_hamiltonian(psi_in.reshape(psi_out.shape),psi_out,J,h)
	return yout.ravel()


class TFIM_1d(object):
	def __init__(self,L,Nf=0):
		self._L = L
		self._psi_out = np.zeros((2*L,L),dtype=np.complex128)

	def dot(self,psi_in,J,h,psi_out=None):
		if psi_out is None:
			psi_out = psi_in.copy()

		_1d_hamiltonian(psi_in,psi_out,J,h)

		return psi_out

	def anneal(self,psi_i,ti,tf,J_func,h_func,**solver_args):
		f_params = (self._psi_out,J_func,h_func)
		return evolve(psi_i,ti,tf,_SE_1d,f_params=f_params,**solver_args)


