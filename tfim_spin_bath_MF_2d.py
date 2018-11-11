from quspin.basis import spin_basis_general,tensor_basis
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
from quspin.basis.transformations import square_lattice_trans
from tilted_square import tilted_square_transformations

import numpy as np
import cProfile,os,sys,time
import matplotlib.pyplot as plt




def anneal_bath_1(n,m,Nb,T,gamma=0.2,omega=1.0,path="."):
	ti = time.time()
	n,m = min(n,m),max(n,m)
	filename = os.path.join(path,"spin_bath_exact_n_{}_m_{}_Nb_{}_T_{}_gamma_{}_omega_{}.npz".format(n,m,Nb,T,gamma,omega))
	if os.path.isfile(filename):
		print "file_exists...exiting run."
		exit()

	if Nb%2 == 1:
		S = "{}/2".format(Nb)
	else:
		S = "{}".format(Nb//2)

	print "creating basis"
	bath_basis = spin_basis_general(1,S=S)
	N = n**2+m**2

	Ns_block_est = max((2**N)/(N),1000)

	if n!= 0:
		T1,t1,T2,t2,Pr,pr,Tx,Ty = tilted_square_transformations(n,m)
		blocks = dict(tx=(Tx,0),ty=(Ty,0),pb=(Pr,0))
		spin_basis = spin_basis_general(N,S="1/2",pauli=True,Ns_block_est=Ns_block_est,**blocks)
	else:
		L = m
		tr = square_lattice_trans(L,L)

		Tx = tr.T_x
		Ty = tr.T_y
		
		blocks = dict(tx=(Tx,0),ty=(Ty,0),px=(tr.P_x,0),py=(tr.P_y,0),pd=(tr.P_d,0))
		spin_basis = spin_basis_general(N,S="1/2",pauli=True,Ns_block_est=Ns_block_est,**blocks)

	basis = tensor_basis(spin_basis,bath_basis)
	print "n,m={},{}, H-space size: {}".format(n,m,basis.Ns)


	J_list = [[-1.0,i,Tx[i]] for i in range(N)]
	J_list.extend([-1.0,i,Ty[i]] for i in range(N))
	h_list = [[-1.0,i] for i in range(N)]

	bath_energy=[[omega/Nb,0]]
	SB_xy_list = [[gamma/(4.0*Nb),i,0] for i in range(N)]
	SB_zz_list = [[gamma/(2.0*Nb),i,0] for i in range(N)]


	A = lambda t:(t/T)**2
	B = lambda t:(1-t/T)**2

	static = [
			  ["|z",bath_energy],
			 ]
	dynamic = [["zz|",J_list,A,()],
			   ["x|",h_list,B,()],
			   ["+|-",SB_xy_list,B,()],
			   ["-|+",SB_xy_list,B,()],
			   ["z|z",SB_zz_list,B,()],
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




n = int(sys.argv[1])
m = int(sys.argv[2])
Nb = int(sys.argv[3])
T = float(sys.argv[4])
gamma = float(sys.argv[5])
omega = float(sys.argv[6])
model = int(sys.argv[7])
path = sys.argv[8]

if model == 1:
	anneal_bath_1(n,m,Nb,T,gamma,omega,path)

