from quspin.basis import spin_basis_1d,tensor_basis
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
import numpy as np
import cProfile,os,sys
import matplotlib.pyplot as plt



def anneal_bath(L,T,gamma=0.2,omega=1.0,path="."):
	filename = os.path.join(path,"spin_bath_exact_L_{}_T_{}_gamma_{}_omega_{}.npz".format(L,T,gamma,omega))
	if os.path.isfile(filename):
		print "file_exists...exiting run."
		exit()

	if L%2 == 1:
		S = "{}/2".format(L)
	else:
		S = "{}".format(L//2)

	print "creating basis"
	spin_basis = spin_basis_1d(L,pauli=True,kblock=0,pblock=1)
	bath_basis = spin_basis_1d(1,S=S)
	basis = tensor_basis(spin_basis,bath_basis)
	print "L={}, H-space size: {}".format(L,basis.Ns)
	# exit()

	bath_energy=[[omega/L,0]] # photon energy
	SB_list = [[gamma/np.sqrt(L),i,0] for i in range(L)]
	h_list = [[-1,i] for i in range(L)]
	J_list = [[-1,i,(i+1)%L] for i in range(L)]

	static_SB = [["+|-",SB_list],["-|+",SB_list]]
	static = [["|z",bath_energy]]
	dynamic = [["x|",h_list,lambda t:(1-t/T)**2,()],
				["zz|",J_list,lambda t:(t/T)**2,()],]

	print "creating hamiltonian"
	kwargs=dict(basis=basis,dtype=np.float64,
		check_symm=False,check_pcon=False,check_herm=False)
	H_B = hamiltonian(static,[],**kwargs)
	H_S = hamiltonian([],dynamic,**kwargs)
	V =hamiltonian(static_SB,[],**kwargs)

	H = H_B+H_S+V
	print "solving initial state"
	E0,psi_0 = H.eigsh(k=1,which="SA",time=0)
	psi_0 = psi_0.ravel()

	print "evolving"
	out = np.zeros(psi_0.shape,dtype=np.complex128)
	psi_f = evolve(psi_0,0,T,H._hamiltonian__omp_SO,f_params = (out,),solver_name="dop853")

	print "saving"
	np.savez_compressed(filename,psi=psi_f)
	print "dome."


def anneal_bath_2(L,T,gamma=0.01,path="."):
	filename = os.path.join(path,"spin_bath_exact_model2_L_{}_T_{}_gamma_{}.npz".format(L,T,gamma))
	if os.path.isfile(filename):
		print "file_exists...exiting run."
		exit()

	if L%2 == 1:
		S = "{}/2".format(L)
	else:
		S = "{}".format(L//2)

	print "creating basis"
	spin_basis = spin_basis_1d(L,pauli=True,kblock=0,pblock=1)
	bath_basis = spin_basis_1d(1,S=S)
	basis = tensor_basis(spin_basis,bath_basis)
	print "L={}, H-space size: {}".format(L,basis.Ns)
	# exit()

	B_list = [[-1,0]] # photon energy
	SB_list = [[-2*gamma/L,i,0] for i in range(L)]
	h_list = [[-1,i] for i in range(L)]
	J_list = [[-1,i,(i+1)%L] for i in range(L)]


	A = lambda t:(t/T)**2
	B = lambda t:(1-t/T)**2
	dynamic = [["+|",h_list,B,()],["-|",h_list,B,()],
				["|+",B_list,B,()],["|-",B_list,B,()],
				["zz|",J_list,A,()],
				["z|z",SB_list,A,()]]

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
omega = float(sys.argv[4])
model = int(sys.argv[5])
path = sys.argv[6]

if model == 1:
	anneal_bath(L,T,gamma,omega,path)
elif model == 2:
	anneal_bath_2(L,T,gamma,path)