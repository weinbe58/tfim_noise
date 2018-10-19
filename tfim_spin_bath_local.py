from quspin.basis import spin_basis_1d,tensor_basis
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
import numpy as np
import cProfile,os,sys
import matplotlib.pyplot as plt


def anneal_bath(L,T,gamma=0.01,path="."):
	filename = os.path.join(path,"spin_bath_exact_model3_L_{}_T_{}_gamma_{}.npz".format(L,T,gamma))
	if os.path.isfile(filename):
		print "file_exists...exiting run."
		exit()


	print "creating basis"
	basis = spin_basis_1d(2*L,pauli=True,a=2,pblock=1,kblock=0,zblock=1)
	print "L={}, H-space size: {}".format(L,basis.Ns)
	exit()
	J_list = [[-1,i,(i+2)%(2*L)] for i in range(0,2*L,2)]
	J_list += [[-1,i,(i+2)%(2*L)] for i in range(1,2*L,2)]
	J_list += [[-gamma,i,(i+1)] for i in range(0,2*L,2)]

	h_list = [[-1,i] for i in range(2*L)]



	A = lambda t:(t/T)**2
	B = lambda t:(1-t/T)**2
	dynamic = [["zz",J_list,A,()],["x",h_list,B,()]]

	H = hamiltonian([],dynamic,basis=basis,dtype=np.float64)


	print "creating hamiltonian"
	kwargs=dict(basis=basis,dtype=np.float64,
		check_symm=False,check_pcon=False,check_herm=False)
	H = hamiltonian([],dynamic,**kwargs)

	print "solving initial state"
	E0,psi_0 = H.eigsh(k=1,which="SA",time=0)
	psi_0 = psi_0.ravel()

	print "evolving"
	out = np.zeros(psi_0.shape,dtype=np.complex128)
	psi_f = evolve(psi_0,0,T,H._hamiltonian__omp_SO,f_params = (out,),solver_name="dop853")

	print "saving"
	np.savez_compressed(filename,psi=psi_f)
	print "dome."


L = int(sys.argv[1])
T = float(sys.argv[2])
gamma = float(sys.argv[3])
model = int(sys.argv[4])
path = sys.argv[5]

if model == 1:
	anneal_bath(L,T,gamma,path)