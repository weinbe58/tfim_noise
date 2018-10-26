from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
import numpy as np
import cProfile,os,sys,time
import matplotlib.pyplot as plt


def anneal_bath(L,T,gamma=0.01,path="."):
	ti = time.time()
	filename = os.path.join(path,"spin_bath_exact_model4_L_{}_T_{}_gamma_{}.npz".format(L,T,gamma))
	if os.path.isfile(filename):
		print "file_exists...exiting run."
		exit()
	N = 2*L
	Z = -(np.arange(N)+1)
	Tx = (np.arange(L)+1)%L
	Tx = np.hstack((Tx,Tx+L))

	P = np.arange(L)[::-1]
	P = np.hstack((P,P+L))


	print "creating basis"
	basis = spin_basis_general(N,pauli=True,pblk=(P,0),kblk=(Tx,0),zblk=(Z,0))
	print "L={}, H-space size: {}".format(L,basis.Ns)

	J_list = [[-1,i,(i+1)%L] for i in range(L)]
	J_list += [[-gamma,L+i,L+(i+1)%L] for i in range(L)]
	J_list += [[-gamma,i,i+L] for i in range(L)]

	h_list = [[-1,i] for i in range(2*L)]



	A = lambda t:(t/T)**2
	B = lambda t:(1-t/T)**2
	dynamic = [["zz",J_list,A,()],["x",h_list,B,()]]
	kwargs=dict(basis=basis,dtype=np.float64,
		check_symm=False,check_pcon=False,check_herm=False)
	H = hamiltonian([],dynamic,**kwargs)


	print "creating hamiltonian"
	kwargs=dict(basis=basis,dtype=np.float64
		# ,check_symm=False,check_pcon=False,check_herm=False
		)
	H = hamiltonian([],dynamic,**kwargs)

	print "solving initial state"
	E0,psi_0 = H.eigsh(k=1,which="SA",time=0)
	psi_0 = psi_0.ravel()

	print "evolving"
	out = np.zeros(psi_0.shape,dtype=np.complex128)
	psi_f = evolve(psi_0,0,T,H._hamiltonian__omp_SO,f_params = (out,),solver_name="dop853")

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