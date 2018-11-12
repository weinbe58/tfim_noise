import numpy as np
import glob,sys,os 
import matplotlib.pyplot as plt
from itertools import product
from quspin.basis import spin_basis_general,tensor_basis
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
from quspin.basis.transformations import square_lattice_trans
from tilted_square import tilted_square_transformations

import numpy as np
import cProfile,os,sys
from six import itervalues

def get_filedict(filename):
	filename = os.path.split(filename)[-1] # remove path
	filename = ".".join(filename.split(".")[:-1]) # rm extension
	filelist = filename.split("_")

	n = int(filelist[filelist.index("n")+1])
	filelist.pop(filelist.index("n")+1)
	filelist.pop(filelist.index("n"))

	m = int(filelist[filelist.index("m")+1])
	filelist.pop(filelist.index("m")+1)
	filelist.pop(filelist.index("m"))

	filedict = {"size":(n,m)}
	for i,item in enumerate(filelist):
		if item.replace(".","").isdigit():
			filedict[filelist[i-1]] = eval(item)

	return filedict



def get_operators(size,Nb):
	n,m = size
	if Nb%2 == 1:
		S = "{}/2".format(Nb)
	else:
		S = "{}".format(Nb//2)

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
	J_list = [[-1.0,i,Tx[i]] for i in range(N)]
	J_list.extend([-1.0,i,Ty[i]] for i in range(N))

	M_list = [[1.0/N**2,i,i] for i in range(N)]
	M_list += [[2.0/N**2,i,j] for i in range(N) for j in range(N) if i > j]

	kwargs=dict(basis=basis,dtype=np.float64,
		check_symm=False,check_pcon=False,check_herm=False)
	print size
	H_S = hamiltonian([["zz|",J_list]],[],**kwargs)
	M2 = hamiltonian([["zz|",M_list]],[],**kwargs)

	return H_S,M2

def process_file(filename,H_S,M2):
	N = H_S.basis.basis_left.N
	file_data = np.load(filename)
	psi = file_data["psi"]
	return H_S.expt_value(psi).real+2*N,M2.expt_value(psi).real




filelist = glob.glob(sys.argv[1])
filelist.sort()

data_dict = {}
list_dict = {}
oper_dict = {}

oper_keys = ["size","Nb"]

print filelist

def sorter(filename):
	filedict = get_filedict(filename)
	return filedict["size"][0]**2+filedict["size"][1]**2


filelist.sort(key=sorter)

if filelist:
	for filename in filelist:
		print filename 	
		filedict = get_filedict(filename)

		oper_key = tuple(filedict[key] for key in oper_keys)

		if oper_key not in oper_dict:
			oper_dict[oper_key] = get_operators(*oper_key)

		for key,value in filedict.items():
			if key in list_dict:
				list_dict[key].add(value)
			else:
				list_dict[key] = set([value])




		key = tuple(filedict.items())
		data_dict[(key,oper_key)] = filename

	keys = list_dict.keys()
	keys.sort()
	print keys
	shape = ()
	for key in keys:
		try:
			np_list = np.fromiter(list_dict[key],dtype=np.float)
			np_list.sort()
		except ValueError:
			py_list = list(list_dict[key])
			py_list.sort(key=lambda x:x[0]**2+x[1]**2)
			np_list = np.empty(len(py_list),dtype=object)
			np_list[:] = py_list

		
		list_dict[key] = np_list
		shape = shape + np_list.shape


	n_op = max([len(val) for val in itervalues(oper_dict)])

	shape = shape + (n_op,)
	data = np.full(shape,np.nan,dtype=np.float)
	print data.shape
	for (key,oper_key),filename in data_dict.items():
		key_dict = dict(key)
		index = []
		for file_key in keys:
			m = [m==key_dict[file_key] for m in list_dict[file_key]]
			if any(m):
				i = np.argwhere(m)[0,0]
			else:
				raise ValueError

			index.append((i,))
		index.append(Ellipsis)
		data[tuple(index)] = process_file(filename,*oper_dict[oper_key])

	np.savez_compressed(sys.argv[2],data=data,**list_dict)

