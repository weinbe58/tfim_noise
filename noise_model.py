

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

		try:
			N = len(J0)
		except TypeError:
			N = 0
		Nc = len(omega)


		if N==0:
			self._omega = omega
			self._c = np.random.normal(0,width/np.sqrt(Nc),size=(Nc,))+ \
					1j*np.random.normal(0,width/np.sqrt(Nc),size=(Nc,))
		else:
			self._omega = omega
			self._c = np.random.normal(0,width/np.sqrt(Nc),size=(Nc,N))+ \
					1j*np.random.normal(0,width/np.sqrt(Nc),size=(Nc,N))

		self._Nc = Nc

	def _noise(self,t):
		return (np.exp(1j*t*self._omega).dot(self._c)).real


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