from numba import jit
from TFIM_1d_dyn import TFIM_general,get_C_fermion,get_C_spin
from quspin.tools.evolution import evolve
import numpy as np



@jit(nopython=True)
def hamiltonian_1d(yin,yout,J,h):
    L = len(h)
    J[-1] *= -1
    for i in range(L):
        ip = (i+1)%L
        im = (i-1)%L
        yout[i,:]   =  (J[im]*(yin[im,:] - yin[L+im,:]) + J[i]*(yin[ip,:] + yin[L+ip,:]))/2.0  - yin[i,:]*h[i]    
        yout[L+i,:] =  yin[L+i,:]*h[i] - ((J[im]*(yin[L+im,:] - yin[im,:]) + J[i]*(yin[L+ip,:] + yin[ip,:])))/2.0

    J[-1] *= -1

@jit(nopython=True)
def ising_local_1d(yin,J,out):
    L = len(J)
    for i in range(L):
        ip = (i+1)%L
        y0 = y[i]
        y1 = y[ip]
        yL0 = y[L+ip]
        yL1 = y[L+ip]
        me = np.conj(y0) * (y1 + yL1)
        me += np.conj(y1) * (y0 - yL0)
        me -= np.conj(yL0) * (y1 + yL1)
        me += np.conj(yL1) * (y0 - yL0)

@jit("void(c16[:,:],f8[:],f8[:])",nopython=True)
def t_field_local_1d(yin,h,out):
    L = len(h)
    for i in range(L):
    	me = 0.0
    	for j in range(L):
            me += h[i]*(np.abs(yin[L+i,j])**2 - np.abs(yin[i,j])**2)    

        out[i] = me


def SO(t,yin,yout,T,gamma,omegas,work,J_func,h_func,No,L):
    J = J_func(t,T)
    h = h_func(t,T)

    out = np.zeros_like(h)

    eta_in = yin[:L*No].reshape((No,L))
    psi_in = yin[L*No:].reshape((2*L,L))


    eta_out = yout[:L*No].reshape((No,L))
    psi_out = yout[L*No:].reshape((2*L,L))

    t_field_local_1d(psi_in,h,out)

    eta_out[:] = -1j * gamma * out
    np.multiply(omegas,t,out=work)
    np.cos(work,out=work)
    h += gamma * np.einsum("ij,ji->i",work,eta_in.real)
    hamiltonian_1d(psi_in,psi_out,J,h)

    psi_out *= -2j

    return yout




L = 8*8
T = 1000
No = 1000
gamma = 0.01
J0 = -np.ones(L)

J_func = lambda t,T:J0*(t/T)**2
h_func = lambda t,T:np.full(L,-(1-t/T)**2)


eta0 = np.random.normal(0,1,size=(No,L))+1j*np.random.normal(0,1,size=(No,L))
eta0 *= (0.5/np.linalg.norm(eta0,axis=0))

omegas = np.random.uniform(0,1,size=(L,No))
work = np.zeros_like(omegas)


A = lambda x:(1-x)**2
B = lambda x:x**2
s = lambda t:t/T

tfim_H = TFIM_general(L,A,B,s,(),J=J0,h=-1)


E,V = tfim_H.eigh(h=1.0,J=0.0)



psi0 = np.array(V[:,:L],dtype=np.complex128,copy=True)

y0 = np.hstack((eta0.ravel(),psi0.ravel()))
yout = np.zeros_like(y0)

f_params = (yout,T,gamma,omegas,work,J_func,h_func,No,L)
yf = evolve(y0,0,T,SO,f_params=f_params,solver_name="dop853",atol=1.1e-15,rtol=1.1e-15)


etaf = yf[:L*No].reshape((No,L))
psif = yf[L*No:].reshape((2*L,L))


AA,BB,AB,BA = get_C_fermion(L,psif)
m2 = get_C_spin(AA,BB,AB,BA).real.sum()/L**2
q = tfim_H.expt_value(psif,J=1.0,h=0.0).real + L
print m2,q,np.linalg.norm(etaf,axis=0)

