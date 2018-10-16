import numpy as np
import os,sys
from quspin.operators import hamiltonian
from numba import jit,cuda,uint64,prange
from pyculib.blas import Blas
from time import time


@jit(nopython=True,parallel=True)
def ising_term(J,i,j,ising):
    for k in prange(ising.size):
        ising[k] += J*(1 - ((((k>>i)^(k>>j))&1)<<1)) 

@jit(nopython=True,parallel=True)
def z_term(hz,i,field):
    for k in prange(field.size):
        field[k] += hz*(((k>>i)&1)<<1)-1

@jit(nopython=True,parallel=True)
def local_x(yin,yout,i):
    for s in prange(yin.size):
        yout[s] = yin[s^(1<<i)]

@jit(nopython=True,parallel=True)
def local_y(yin,yout,i):
    for s in prange(yin.size):
        yout[s] = 1j*((((s>>i)&1)<<1)-1)*yin[s^(1<<i)]

@jit(nopython=True,parallel=True)
def local_z(yin,yout,i):
    for s in prange(yin.size):
        yout[s] = ((((s>>i)&1)<<1)-1)*yin[s]

@jit(nopython=True,parallel=True)
def H(psi_1,psi_2,H_ising,pg,J,h):
    for i in prange(psi_1.size):
        b = uint64(1) # use this number fo flip bit to get column index
        ME = (1 + pg + J*H_ising[i])*psi_1[i] 
        for j in range(N):
            ME += -h*psi_1[i^b] # x-field action
            b <<= 1 # shift flipping fit to the right

        psi_2[i] = ME

@jit(nopython=True,parallel=True)
def qsd(psi_1,T,N,dt,gamma,H_ising):
    P0 = 1 - 3*gamma*dt*N
    Pg = 3*N*dt*gamma/2.0
    niter = int(T/dt)

    nx = 0
    ny = 0
    nz = 0

    psi_1 = psi_1.astype(np.complex128)
    psi_2 = np.zeros(psi_1.shape,dtype=np.complex128)
    for i in range(niter):
        # print i
        if np.random.ranf() < P0: # evolve with hamiltonian
            J = -1j*dt*(dt*i/T)**2
            h = -1j*dt*(1-dt*i/T)**2
            H(psi_1,psi_2,H_ising,Pg,J,h)
            psi_2 /= np.linalg.norm(psi_2)

        else: # quantum jump
            psi_2[:] = psi_1[:]
            j = np.random.randint(3*N)
            alpha = j%3
            site = j//3
            if alpha == 0:
                nx += 1
                local_x(psi_1,psi_2,site)
            elif alpha == 1:
                ny += 1
                local_y(psi_1,psi_2,site)
            else:
                nz += 1
                local_z(psi_1,psi_2,site)

        psi_1[:] = psi_2[:]


    return psi_1,nx,ny,nz


@cuda.jit
def local_x_cuda(yin,yout,i):
    s = cuda.grid(1)
    if s < yin.size:
        yout[s] = yin[s^(1<<i)]

@cuda.jit
def local_y_cuda(yin,yout,i):
    s = cuda.grid(1)
    if s < yin.size:
        yout[s] = 1j*((((s>>i)&1)<<1)-1)*yin[s^(1<<i)]

@cuda.jit
def local_z_cuda(yin,yout,i):
    s = cuda.grid(1)
    if s < yin.size:
        yout[s] = ((((s>>i)&1)<<1)-1)*yin[s]

@cuda.jit
def H_cuda(yin,yout,H_ising,pg,J,h):
    s = cuda.grid(1)
    if s < yin.size:
        b = uint64(1) # use this number fo flip bit to get column index
        ME = (1 + pg + J * H_ising[s])*yin[s] 
        for j in range(N):
            ME += -h*yin[s^b] # x-field action
            b <<= 1 # shift flipping fit to the right

        yout[s] = ME

@cuda.jit
def assign_array(y1,y2):
    i = cuda.grid(1)
    if i < y1.size:
        y1[i] = y2[i]

@cuda.jit
def divide_array(y1,a):
    i = cuda.grid(1)
    if i < y1.size:
        y1[i] /= a

@jit
def qsd_cuda(psi_1,T,N,dt,gamma,H_ising):
    P0 = max(0,1 - 3*N*gamma*dt)
    Pg = 3*N*dt*gamma/2.0
    niter = int(T/dt)
    nbk = max(psi_1.size//1024,1)
    nth = 1024


    nx = 0
    ny = 0
    nz = 0

    psi_1 = cuda.to_device(psi_1.astype(np.complex128))
    psi_2 = cuda.device_array(psi_1.shape,dtype=np.complex128)
    H_ising = cuda.to_device(H_ising)

    blas = Blas()
    for i in range(niter):
        if np.random.ranf() < P0: # evolve with hamiltonian
            J = -1j*dt*(dt*i/T)**2
            h = -1j*dt*(1-dt*i/T)**2
            H_cuda[nbk,nth](psi_1,psi_2,H_ising,Pg,J,h)
            nrm2 = blas.nrm2(psi_2)
            divide_array[nbk,nth](psi_2,nrm2)
        else: # quantum jump
            j = np.random.randint(3*N)
            alpha = j%3
            site = j//3
            if alpha == 0:
                nx+=1
                local_x_cuda[nbk,nth](psi_1,psi_2,site)
            elif alpha == 1:
                ny+=1
                local_y_cuda[nbk,nth](psi_1,psi_2,site)
            else:
                nz+=1
                local_z_cuda[nbk,nth](psi_1,psi_2,site)

        assign_array[nbk,nth](psi_1,psi_2)


    return psi_1.copy_to_host(),nx,ny,nz


Lx = 4
Ly = 4
N = Lx*Ly
T = 1000
dt = 1e-2
gamma = 1e-3


up_iter = [(i+Lx*j,i+Lx*(j+1)) for i in range(Lx) for j in range(Ly-1)]
left_iter = [(i+Lx*j,(i+1)+Lx*j) for i in range(Lx-1) for j in range(Ly)]


J_list = [[-1,i,j] for i,j in up_iter]
J_list.extend([[-1,i,j] for i,j in left_iter])

M_list = [[1.0,i,j] for i in range(N) for j in range(N)]


H_ising = np.zeros(2**N,dtype=np.int8)
for J,i,j in J_list:
    ising_term(J,i,j,H_ising)



M2 = np.zeros(2**N,dtype=np.float64)
for _,i,j in M_list:
    ising_term(1.0/N**2,i,j,M2)

psi_i = np.ones(2**N)/np.sqrt(2**N)

# if N >= 16 and cuda.is_available():
#     psi_f = qsd_cuda(psi_i,T,N,dt,gamma,H_ising)
# else:
#     psi_f = qsd(psi_i,T,N,dt,gamma,H_ising)


for i in range(1000):
    ti = time()
    psi_f,nx,ny,nz = qsd(psi_i,T,N,dt,gamma,H_ising)
    # psi_f,nx,ny,nz = qsd_cuda(psi_i,T,N,dt,gamma,H_ising)
    print "cpu: {:5.4f} M2: {:5.4f} n: {},{},{}".format(time()-ti, psi_f.conj().dot(M2*psi_f).real,nx,ny,nz)

