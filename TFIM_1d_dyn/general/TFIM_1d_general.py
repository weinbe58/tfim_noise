import scipy.sparse as sp
import numpy as np
from numba import njit



class TFIM_general(object):
	def __init__(self,L,A,B,s,s_args=(),Nf=0,J=-1.0,h=-1.0):
		self.L = L
		self.s = s
		self.A = A
		self.B = B
		self.s_args = s_args

		J = np.asarray(J)
		h = np.asarray(h)

		if J.ndim == 0:
			J = J*np.ones(L)

		if h.ndim == 0:
			h = h*np.ones(L)

		if len(h) != L:
			raise ValeuError("h must be scalar or an array of length L")

		if len(J) != L:
			raise ValeuError("J must be scalar or an array of length L")


		data = np.zeros((4,L),dtype=J.dtype)

		data[0,0] = J[-1]*(-1)**(Nf+1)
		data[1,:-1] = J[:-1]
		data[2,1:] = J[:-1]
		data[3,-1] = J[-1]*(-1)**(Nf+1)

		shift = np.array([1-L,-1,1,L-1])
		self._A = sp.dia_matrix((data,shift),shape=(L,L))/2.0

		data[:2,:]= -data[:2,:]
		data[0,0] = -data[0,0]
		data[3,-1] = -data[3,-1]

		self._B = sp.dia_matrix((data,shift),shape=(L,L))/2.0
		self._h = sp.dia_matrix((h,[0]),shape=(self.L,self.L))
		# print A.todense()
		# print B.todense()
		# exit()

		self._Hzz = sp.bmat([[self._A,self._B],[-self._B,-self._A]],format="csr")
		self._Hx = np.hstack((h,-h))



	@property
	def shape(self):
		return (2*self.L,2*self.L)

	def set_s(self,s,s_args):
		self.s = s
		self.s_args = s_args

	def __call__(self,J=1.0,h=0.0,time=None):
		if time is not None:
			s = self.s(time,*self.s_args)
			h = self.A(s)
			J = self.B(s)

		return J*self._Hzz + h*sp.dia_matrix((self._Hx,[0]),shape=(2*self.L,2*self.L))

	def eigh(self,J=1.0,h=0.0,time=None):
		H = self.__call__(J=J,h=h,time=time).toarray()
		return np.linalg.eigh(H)

	def dot(self,v,J=1.0,h=0.0,time=None):
		if time is not None:
			s = self.s(time,*self.s_args)
			h = self.A(s)
			J = self.B(s)
		v_f = J*self._Hzz.dot(v)
		v_f += np.einsum(",i,ij->ij",h,self._Hx,v)
		return v_f

	def expt_value(self,y,**dot_args):
		y_right = self.dot(y,**dot_args)
		return np.einsum("ij,ij->",y.conj(),y_right)

	def SO(self,t,y):
		y = y.reshape((2*self.L,-1))
		return -2j*(self.dot(y,time=t).ravel())


	def SO_Z(self,t,y):
		s = self.s(time,*self.s_args)
		h = self.A(s)
		J = self.B(s)
		y = y.reshape((self.L,self.L))
		y_out = self._A * y + self._A.dot(y.T).T
		y_out += self._B + y.dot(self._B.dot(y))
		y_out *= J
		y_out += h * (self._h.dot(y) + self._A.dot(y.T).T)
		return -2j*y_out.ravel()


def get_C_fermion(L,psi):
	g = psi[:L]
	h = psi[L:].conj()

	phi = (h.conj()+g)
	psi = (h.conj()-g)

	AA =  phi.dot(phi.T.conj())
	BB = -psi.dot(psi.T.conj())

	AB = -phi.dot(psi.T.conj())
	BA =  psi.dot(phi.T.conj())

	return AA,BB,AB,BA



# def get_C_spin(AA,BB,AB,BA):
# 	L = AA.shape[0]
# 	C = np.zeros((L,L),dtype=np.complex128)
# 	gamma = np.zeros((L-1,2*L,2*L),dtype=np.complex128)
# 	A = np.zeros_like(gamma)
# 	sy = np.zeros((2,2),dtype=np.complex128)
# 	sy[0,1] = -1j
# 	sy[1,0] =  1j


# 	for n in range(1,L):
# 		K = np.kron(np.identity(n),sy)
# 		gamma_v = gamma[:L-n,:2*n,:2*n]
# 		A_v = A[:L-n,:2*n,:2*n]
# 		gamma_v[...] = 0.0
# 		get_gamma_n(L-n,n,gamma_v,AA,BB,AB,BA)

# 		np.matmul(K,gamma_v,out=A_v)
# 		e = np.linalg.eigvals(A_v)

# 		e_tr = np.log(e).sum(axis=-1)
# 		e_tr = ((1j)**n)*np.exp(e_tr/2.0)
# 		for i in range(L-n):
# 			C[i,i+n] = e_tr[i]


# 	C += C.T

# 	for i in range(L):
# 		C[i,i] = 1.0

# 	return C

@njit
def get_gamma_n(ni,n,gamma,AA,BB,AB,BA):
	for i in range(ni):
		for l in range(n):
			for k in range(n):
				gamma[i,2*l  ,2*k  ] = BB[i+l  ,i+k  ]
				gamma[i,2*l+1,2*k  ] = AB[i+l+1,i+k  ]
				gamma[i,2*l  ,2*k+1] = BA[i+l  ,i+k+1]
				gamma[i,2*l+1,2*k+1] = AA[i+l+1,i+k+1]

				if l==k:
					gamma[i,2*l  ,2*k  ] += 1
					gamma[i,2*l+1,2*k+1] -= 1


@njit
def set_C(a,L,n,ni,C,e_tr):
	for i in range(ni):
		for j in range(0,L-n-i,a):
			C[j+i,j+i+n] = e_tr[i]


def get_C_spin(AA,BB,AB,BA,a=None):
	L = AA.shape[0]

	if a is None:
		a = L
	else:
		if type(a) is not int:
			raise TypeError("a must be integer.")

		if L%a != 0:
			raise ValueError("number of total sites must be multiple of unit cell size.")

	C = np.zeros((L,L),dtype=np.complex128)
	gamma = np.zeros((a,2*L,2*L),dtype=np.complex128)
	sy = np.zeros((2,2),dtype=np.complex128)
	sy[0,1] = -1j
	sy[1,0] = 1j

	for n in range(1,L):
		ni = min(a,L-n)
		K = np.kron(np.identity(n),sy)
		gamma_v = gamma[:ni,:2*n,:2*n]
		gamma_v[...] = 0.0
		get_gamma_n(ni,n,gamma_v,AA,BB,AB,BA)

		A = np.matmul(K,gamma_v)
		e = np.linalg.eigvals(A)
		e_tr = np.log(e).sum(axis=-1)
		e_tr = ((1j)**n)*np.exp(e_tr/2.0)
		set_C(a,L,n,ni,C,e_tr)

	C += C.T
	np.fill_diagonal(C,1.0)

	return C
