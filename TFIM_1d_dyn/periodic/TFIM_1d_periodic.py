import scipy.sparse as sp
from numpy import sin,cos,pi
import numpy as np




class TFIM(object):
	# based on TFIM solution in: https://arxiv.org/abs/1207.5242
	def __init__(self,L,A,B,s,s_args=()):
		self.L=L
		self.s = s
		self.A = A
		self.B = B
		self.s_args = s_args
		k_points  = pi*np.arange(1,L,2)/L


		self.omega_k = 2*cos(k_points)
		self.delta_k = 2*sin(k_points)

		H_zz = np.zeros((self.L//2,2,2))
		H_zz[:,0,0] = -self.omega_k
		H_zz[:,1,1] = self.omega_k
		H_zz[:,0,1] = self.delta_k
		H_zz[:,1,0] = self.delta_k

		self.Hzz = sp.block_diag(H_zz,format="csr")
		self.y_out = np.zeros((L,),dtype=np.complex128)

	@property
	def shape(self):
		return (self.L,self.L)

	def set_s(self,s,s_args):
		self.s = s
		self.s_args = s_args

	def eigh(self,h=1,J=1,time=None):
		if time is not None:
			s = self.s(time,*self.s_args)
			h = self.A(s)
			J = self.B(s)

		H_list = np.zeros((self.L//2,2,2))
		H_list[:,0,0] =  2*h-J*self.omega_k
		H_list[:,1,1] = -2*h+J*self.omega_k 
		H_list[:,0,1] =  J*self.delta_k
		H_list[:,1,0] =  J*self.delta_k

		return np.linalg.eigh(H_list)

	def dot(self,y,h=1,J=1,time=None):
		if time is not None:
			s = self.s(time,*self.s_args)
			h = self.A(s)
			J = self.B(s)


		y_out = J*self.Hzz.dot(y)
		y_out[0::2] +=  2*h*y[0::2]
		y_out[1::2] += -2*h*y[1::2]

		return y_out

	def expt_value(self,y,**dot_args):
		y_right = self.dot(y,**dot_args)
		return y.conj().dot(y_right)

	def SO(self,t,y):
		return -1j*self.dot(y,time=t)

	def ISO(self,t,y):
		s = self.s(t,*self.s_args)
		h = self.A(s)
		J = self.B(s)
		d = np.sqrt((J**2)*(self.delta_k**2)+(2*h-J*self.omega_k)**2)

		y_out = self.dot(y,time=t)
		y_out [0::2] += d*y[0::2]
		y_out [1::2] += d*y[1::2]
	 	if np.linalg.norm(y_out)**2 > 1000:
			print "overflow!!!",t,np.linalg.norm(y_out)**2
		return -y_out
