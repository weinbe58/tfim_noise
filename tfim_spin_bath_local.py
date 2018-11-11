from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
import numpy as np
import cProfile,os,sys,time
import matplotlib.pyplot as plt


def anneal_bath(L,T,gamma=0.01,path="."):
	ti = time.time()
	filename = os.path.join(path,"spin_bath_exact_L_{}_T_{}_gamma_{}.npz".format(L,T,gamma))
	if os.path.isfile(filename):
		print "file_exists...exiting run."
		exit()
	N = 2*L

	Tx = (np.arange(L)+1)%L
	Tx = np.hstack((Tx,Tx+L))

	P = np.arange(L)[::-1]
	P = np.hstack((P,P+L))

	print "creating basis"
	basis = spin_basis_general(N,pblk=(P,0),kblk=(Tx,0))
	print "L={}, H-space size: {}".format(L,basis.Ns)

	Jzz_list = [[-1,i,(i+1)%L] for i in range(L)]
	hx_list = [[-1,i] for i in range(L)]

	hop_bath_list = [[-1.0,L+i,L+(i+1)%L] for i in range(L)]
	hop_bath_list += [[-1.0,L+i,L+(i+1)%L] for i in range(L)]
	int_2_list = [[1.0,L+i,L+(i+1)%L] for i in range(L)]
	int_2_list += [[1.0,L+i,L+(i+2)%L] for i in range(L)]
	int_1_list = [[2.0,L+i] for i in range(L)]


	bath_sys_list = [[gamma,i,i+L] for i in range(L)]

	Jb_list = [[gamma,i,L+i] for i in range(L)]

	A = lambda t:(t/T)**2
	B = lambda t:(1-t/T)**2

	static = [
				["+-",hop_bath_list],
				["-+",hop_bath_list],
				["zz",int_2_list],
				["z",int_1_list]
				]
	dynamic = [
				["zz",Jzz_list,A,()],
				["x",hx_list,B,()],
				["+-",bath_sys_list,B,()],
				["-+",bath_sys_list,B,()],
				]

	kwargs=dict(basis=basis,dtype=np.float64,
		check_symm=False,check_pcon=False,check_herm=False)
	H = hamiltonian(static,dynamic,**kwargs)


	print "creating hamiltonian"
	kwargs=dict(basis=basis,dtype=np.float64
		,check_symm=False,check_pcon=False,check_herm=False
		)
	H = hamiltonian([],dynamic,**kwargs)

	print "solving initial state"
	E0,psi_0 = H.eigsh(k=1,which="SA",time=0)
	psi_0 = psi_0.ravel()

	print "evolving"
	out = np.zeros(psi_0.shape,dtype=np.complex128)
	psi_f = evolve(psi_0,0,T,H._hamiltonian__omp_SO,f_params = (out,),
		solver_name="dop853",atol=1.1e-15,rtol=1.1e-15)

	psi_f /= np.linalg.norm(psi_f)

	print "saving"
	np.savez_compressed(filename,psi=psi_f)
	print "dome.... {} sec".format(time.time()-ti)


L = int(sys.argv[1])
T = float(sys.argv[2])
gamma = float(sys.argv[3])
model = int(sys.argv[4])
path = sys.argv[5]

if model == 1:
	anneal_bath(L,T,gamma,path)
