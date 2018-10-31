from quspin.basis import spin_basis_1d,tensor_basis,boson_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
import numpy as np
import cProfile,os,sys,time
import matplotlib.pyplot as plt



def plot(L,Nb,gamma,omega):
	if Nb%2 == 1:
		S = "{}/2".format(Nb)
	else:
		S = "{}".format(Nb//2)

	print "creating basis"
	spin_basis = spin_basis_1d(L,pauli=True,kblock=0,pblock=1)
	bath_basis = spin_basis_1d(1,S=S)
	basis = tensor_basis(spin_basis,bath_basis)
	print "L={}, H-space size: {}".format(L,basis.Ns)

	bath_energy_2=[[-omega/Nb,0,0]]
	SB_1_list = [[-gamma/Nb,i,0] for i in range(L)]
	SB_2_list = [[gamma/np.sqrt(Nb),i,0] for i in range(L)]
	B_h_list = [[-1,0]]
	h_list = [[-1,i] for i in range(L)]
	hz_list = [[0.01,i] for i in range(L)]
	J_list = [[-1,i,(i+1)%L] for i in range(L)]

	A = lambda t:(t)**2
	B = lambda t:(1-t)**2

	# static = [
	# 			["+|-",SB_2_list],
	# 			["-|+",SB_2_list],
	# ]
	# dynamic = [
	# 			["x|",h_list,B,()],
	# 			["|+",B_h_list,B,()],
	# 			["|-",B_h_list,B,()],
	# 			["zz|",J_list,A,()],
	# 			["z|z",SB_1_list,A,()],
	# 			["|zz",bath_energy,A,()],
	# ]

	static = [
	]
	dynamic = [
				["x|",h_list,B,()],
				["|+",B_h_list,B,()],
				["|-",B_h_list,B,()],
				["zz|",J_list,A,()],
				["z|z",SB_1_list,A,()],
				["|zz",bath_energy_2,A,()],
	]

	print "creating hamiltonian"
	kwargs=dict(basis=basis,dtype=np.float64,
		check_symm=False,check_pcon=False,check_herm=False)
	H = hamiltonian(static,dynamic,**kwargs)


	times = np.linspace(0,1,301)[1:-1]
	gaps=[]
	for t in times:
		E,V = H.eigsh(k=150,which="SA",time=t,maxiter=100000)
		gaps.append(E-E[0])

	plt.plot(times,gaps,label="L={}".format(L),color="blue")
	plt.show()



for L in [8,10,12]:
	plot(L,L,1,0.01)

plt.legend()
plt.show()



