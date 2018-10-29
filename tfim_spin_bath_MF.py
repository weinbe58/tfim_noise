from quspin.basis import spin_basis_1d,tensor_basis,boson_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
import numpy as np
import cProfile,os,sys,time
import matplotlib.pyplot as plt



def anneal_bath_1(L,Nb,T,gamma=0.2,omega=1.0,path="."):
	ti = time.time()
	filename = os.path.join(path,"spin_bath_exact_L_{}_Nb_{}_T_{}_gamma_{}_omega_{}.npz".format(L,Nb,T,gamma,omega))
	if os.path.isfile(filename):
		print "file_exists...exiting run."
		exit()

	if Nb%2 == 1:
		S = "{}/2".format(Nb)
	else:
		S = "{}".format(Nb//2)

	print "creating basis"
	spin_basis = spin_basis_1d(L,pauli=True,kblock=0,pblock=1)
	bath_basis = spin_basis_1d(1,S=S)
	basis = tensor_basis(spin_basis,bath_basis)
	print "L={}, H-space size: {}".format(L,basis.Ns)

	bath_energy=[[omega/(2*Nb),0]]
	SB_list = [[gamma/np.sqrt(Nb),i,0] for i in range(L)]
	h_list = [[-1,i] for i in range(L)]
	J_list = [[-1,i,(i+1)%L] for i in range(L)]

	A = lambda t:(t/T)**2
	B = lambda t:(1-t/T)**2
	C = lambda t:1.0
	static = [
			  ["|z",bath_energy],
			  ["+|-",SB_list],
			  ["-|+",SB_list]
			 ]
	dynamic = [
			   ["x|",h_list,B,()],
			   ["zz|",J_list,A,()],
			  ]

	print "creating hamiltonian"
	kwargs=dict(basis=basis,dtype=np.float64,
		check_symm=False,check_pcon=False,check_herm=False)
	H = hamiltonian(static,dynamic,**kwargs)

	print "solving initial state"
	E0,psi_0 = H.eigsh(k=1,which="SA",time=0)
	psi_0 = psi_0.ravel()

	print "evolving"
	out = np.zeros(psi_0.shape,dtype=np.complex128)
	psi_f = evolve(psi_0,0,T,H._hamiltonian__omp_SO,f_params = (out,),solver_name="dop853")

	print "saving"
	np.savez_compressed(filename,psi=psi_f)
	print "dome......{} sec".format(time.time()-ti)


def anneal_bath_2(L,T,gamma=0.2,omega=1.0,path="."):
	ti = time.time()
	Nb = 2*L
	filename = os.path.join(path,"spin_bath_exact_L_{}_Nb_{}_T_{}_gamma_{}_omega_{}.npz".format(L,Nb,T,gamma,omega))
	if os.path.isfile(filename):
		print "file_exists...exiting run."
		exit()

	print "creating basis"
	spin_basis = spin_basis_1d(L,pauli=True,kblock=0,pblock=1)
	bath_basis = boson_basis_1d(1,sps=Nb+1)
	basis = tensor_basis(spin_basis,bath_basis)
	print "L={}, H-space size: {}".format(L,basis.Ns)

	bath_energy=[[omega,0]]
	SB_x_list = [[-gamma/np.sqrt(Nb),i,0] for i in range(L)]
	SB_zz_list = [[-gamma/np.sqrt(Nb),i,(i+1)%L,0] for i in range(L)]
	h_list = [[-1,i] for i in range(L)]
	J_list = [[-1,i,(i+1)%L] for i in range(L)]

	A = lambda t:(t/T)**2
	B = lambda t:(1-t/T)**2
	C = lambda t:1.0
	static = [["|n",bath_energy]]
	dynamic = [["x|",h_list,B,()],
				["zz|",J_list,A,()],
				["x|-",SB_x_list,C,()],
				["x|+",SB_x_list,C,()],
				# ["zz|+",SB_zz_list,A,()],
				# ["zz|-",SB_zz_list,A,()],
				]

	print "creating hamiltonian"
	kwargs=dict(basis=basis,dtype=np.float64,
		check_symm=False,check_pcon=False,check_herm=False)
	H = hamiltonian(static,dynamic,**kwargs)

	print "solving initial state"
	E0,psi_0 = H.eigsh(k=1,which="SA",time=0)
	psi_0 = psi_0.ravel()

	print "evolving"
	out = np.zeros(psi_0.shape,dtype=np.complex128)
	psi_f = evolve(psi_0,0,T,H._hamiltonian__omp_SO,f_params = (out,),solver_name="dop853")

	print "saving"
	np.savez_compressed(filename,psi=psi_f)
	print "dome......{} sec".format(time.time()-ti)


L = int(sys.argv[1])
Nb = int(sys.argv[2])
T = float(sys.argv[3])
gamma = float(sys.argv[4])
omega = float(sys.argv[5])
model = int(sys.argv[6])
path = sys.argv[7]

if model == 1:
	anneal_bath_1(L,Nb,T,gamma,omega,path)
elif model == 2:
	anneal_bath_2(L,T,gamma,omega,path)