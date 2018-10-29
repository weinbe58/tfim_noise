import numpy as np
import glob,sys,os 
import matplotlib.pyplot as plt
from itertools import product
from quspin.basis import spin_basis_1d,tensor_basis
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
import numpy as np
import cProfile,os,sys
import matplotlib.pyplot as plt
from six import itervalues

def get_filedict(filename):
	filename = os.path.split(filename)[-1] # remove path
	filename = ".".join(filename.split(".")[:-1]) # rm extension
	filelist = filename.split("_")

	filedict = {}
	for i,item in enumerate(filelist):
		if item.replace(".","").isdigit():
			filedict[filelist[i-1]] = eval(item)

	return filedict



def get_operators(L,Nb):
	if Nb%2 == 1:
		S = "{}/2".format(Nb)
	else:
		S = "{}".format(Nb//2)

	spin_basis = spin_basis_1d(L,pauli=True,kblock=0,pblock=1)
	bath_basis = spin_basis_1d(1,S=S)
	basis = tensor_basis(spin_basis,bath_basis)
	J_list = [[-1,i,(i+1)%L] for i in range(L)]
	M_list = [[1.0/L**2,i,j] for i in range(L) for j in range(L)]

	kwargs=dict(basis=basis,dtype=np.float64,
		check_symm=False,check_pcon=False,check_herm=False)
	print basis.basis_left.L
	H_S = hamiltonian([["zz|",J_list]],[],**kwargs)
	M2 = hamiltonian([["zz|",M_list]],[],**kwargs)

	return H_S,M2

def process_file(filename,H_S,M2):
	L = H_S.basis.basis_left.L
	file_data = np.load(filename)
	psi = file_data["psi"]
	return H_S.expt_value(psi).real+L,M2.expt_value(psi).real




filelist = glob.glob(sys.argv[1])
filelist.sort()

data_dict = {}
list_dict = {}
oper_dict = {}

oper_keys = ["L","Nb"]

print filelist

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
		np_list = np.fromiter(list_dict[key],dtype=np.float)
		np_list.sort()
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
			i = np.searchsorted(list_dict[file_key],key_dict[file_key])
			index.append((i,))
		index.append(Ellipsis)
		data[tuple(index)] = process_file(filename,*oper_dict[oper_key])

	np.savez_compressed(sys.argv[2],data=data,**list_dict)

