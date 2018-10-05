import numpy as np

class noise(object):
	def __init__(self,J0,ramp,ramp_args=()):
		self._J0 = J0
		self._ramp = ramp
		self._ramp_args = ramp_args

	def __call__(self,t):
		return (self._J0*self._ramp(t,*self._ramp_args)+self._noise(t))


class fourier_noise(noise):
	def __init__(self,J0,ramp,width,omega,ramp_args=()):
		noise.__init__(self,J0,ramp,ramp_args)
		Nc = len(omega)
		try:
			shape = (2*Nc,len(J0))
		except TypeError:
			shape = (2*Nc,)



		self._omega = omega
		self._c =np.random.normal(0,width/np.sqrt(Nc),size=shape)
		self._e = np.zeros(omega.shape,dtype=np.complex128)
		self._e_real = self._e.view(np.float64)

		self._Nc = Nc

	def _noise(self,t):
		np.exp(1j*t*self._omega,out=self._e)
		return self._e_real.dot(self._c)


class osc_noise(noise):
	def __init__(self,J0,ramp,width,q0,Tmax,omega=0.6666666666,A=0.6666666666,gamma=0.1,k=1.0,s=1.0,ramp_args=()):
		noise.__init__(self,J0,ramp,ramp_args)


		self._width=width
		self._omega = omega/s
		self._A = A/s**2
		self._gamma = gamma/s
		self._k = k/s**2

		q0.imag /= s
		try:
			self._N = len(J0)
		except TypeError:
			self._N = 0
			q0 = np.reshape(q0,(1,))
		
		self._q = solve_ivp(self._eom,(0,Tmax),q0,dense_output=True,vectorized=True).sol


	def _noise(self,t):
		if self._N == 0:
			return (self._width*(self._q(t).real%(2*np.pi)-np.pi)/np.pi)[0]
		else:
			return self._width*(self._q(t).real%(2*np.pi)-np.pi)/np.pi



	def _eom(self,t,q):
		return (q.imag)+1j*(-self._gamma*q.imag-self._k*np.sin(q.real)+self._A*np.cos(self._omega*t))