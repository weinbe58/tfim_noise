import math
import numpy as np
from numba import jit,uint64,prange
from scipy.integrate import ode
import os,sys


@jit(nopython=True)
def ising_term(J,i,j,ising):
    for k in range(ising.size):
        ising[k] += J*(1 - ((((k>>i)^(k>>j))&1)<<1)) 

@jit(nopython=True)
def z_term(hz,i,field):
    for k in range(field.size):
        field[k] += hz*(((k>>i)&1)<<1)-1

@jit(nopython=True)
def pauli_local(yin,yout,alpha,site):
    if alpha == 0:
        for s in range(yin.size):
            yout[s] = yin[s^(1<<site)]
    elif alpha == 1:
        for s in range(yin.size):
            yout[s] = 1j*((((s>>site)&1)<<1)-1)*yin[s^(1<<site)]
    else:
        for s in range(yin.size):
            yout[s] = ((((s>>site)&1)<<1)-1)*yin[s]


@jit(nopython=True,parallel=True)
def H(t,psi_in,psi_out,H_ising,T,pg):
    # print np.linalg.norm(psi_in)
    J = (t/T)**2
    h = (1-t/T)**2
    r = (1-t/T)**2
    for s in prange(psi_in.size):
        b = uint64(1) # use this number fo flip bit to get column index
        ME = (-1j*pg*r + J*H_ising[s])*psi_in[s] 
        for j in range(N):
            ME += -h*psi_in[s^b] # x-field action
            b <<= 1 # shift flipping fit to the right

        psi_out[s] = -1j*ME

    return psi_out

def H_wrap(t,psi_in,psi_out,H_ising,T,pg):
    return H(t,psi_in.view(np.complex128),psi_out,H_ising,T,pg).view(np.float64)


def generate_process(gamma,N,tmax):

    def nextTime(rateParameter):
        return -math.log(1.0 - np.random.ranf()) / rateParameter

    rateParameter = gamma.sum()
    process = []
    t = 0
    while True:
        t += nextTime(rateParameter)

        if t > tmax:
            break

        if np.random.ranf() <= (1-t/T)**2:
            p = gamma/gamma.sum()
            i = np.random.choice(np.arange(3*N),p=p.ravel())
            site = i//3
            alpha = i%3
            process.append((t,alpha,site))

    return process


def poisson_ramp(N,T,H_ising,gamma_arr,process):
    pg = gamma_arr.sum()

    psi0 = np.ones(2**N,dtype=np.complex128)/np.sqrt(2**N)
    psi_out = np.zeros_like(psi0)

    solver = ode(H_wrap)
    solver.set_integrator("dop853",nsteps=np.iinfo(np.int32).max,atol=1.1e-7,rtol=1.1e-3)
    solver.set_f_params(psi_out,H_ising,T,pg)

    solver.set_initial_value(psi0.view(np.float64))


    for t,alpha,site in process:
        nt = 10*int(t-solver.t)
        tt = np.linspace(solver.t,t,nt+2)

        for t in tt[1:]:
            solver.integrate(t)

        if solver.successful():
            psi_out[:] = solver.y.view(np.complex128)
            psi_out /= np.linalg.norm(psi_out)
            pauli_local(psi_out,solver.y.view(np.complex128),alpha,site)
        else:
            raise RuntimeError("error code {}".format(solver.get_return_code()))

    nt = 10*int(T-solver.t)
    tt = np.linspace(solver.t,T,nt+2)
    for t in tt[1:]:
        solver.integrate(t)

    if solver.successful():
        psi_out[:] = solver.y.view(np.complex128)
        psi_out /= np.linalg.norm(psi_out)
        return psi_out
    else:
        raise RuntimeError("error code {}".format(solver.get_return_code()))



Lx = int(sys.argv[1])
Ly = 1
N = Lx*Ly
Nb = Lx-1
T = float(sys.argv[2])
gamma = float(sys.argv[3])
N_anneal = int(sys.argv[4])
path = sys.argv[5]

gamma_arr = np.full((Lx,3),gamma,dtype=np.float64)
# gamma_arr[:,:2] = 0


up_iter = [(i+Lx*j,i+Lx*(j+1)) for i in range(Lx) for j in range(Ly-1)]
left_iter = [(i+Lx*j,(i+1)+Lx*j) for i in range(Lx-1) for j in range(Ly)]


J_list = [[-1,i,j] for i,j in up_iter]
J_list.extend([[-1,i,j] for i,j in left_iter])

M_list = [[1.0/N**2,i,j] for i in range(N) for j in range(N)]


H_ising = np.zeros(2**N,dtype=np.int8)
for J,i,j in J_list:
    ising_term(J,i,j,H_ising)

M2 = np.zeros(2**N,dtype=np.float64)
for C,i,j in M_list:
    ising_term(C,i,j,M2)

psi_f_np = poisson_ramp(N,T,H_ising,gamma_arr,())

filename = os.path.join(path,"anneal_err1_noint_L_{}_T_{}_gamma_{}.dat".format(Lx,T,gamma))

with open(filename,"a") as IO:
    for i in range(N_anneal):
        process = generate_process(gamma_arr,N,T)
        print i+1,len(process)
        if len(process) > 0:
            while True:
                try:
                    psi_f = poisson_ramp(N,T,H_ising,gamma_arr,process)
                    break
                except RuntimeError:
                    continue
        else:
            psi_f = psi_f_np

        m2 = np.vdot(psi_f,M2*psi_f).real
        q = np.vdot(psi_f,H_ising*psi_f).real + Nb

        print m2,q
        # IO.write("{:30.15e} {:30.15e}\n".format(q,m2))
        # IO.flush()

