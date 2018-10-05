import numpy as np
import glob,sys,os 
import matplotlib.pyplot as plt
from itertools import product


def bootstrap_mean(Os,n_bs=100):
	Os = np.asarray(Os)
	avg = np.nanmean(Os,axis=0)
	n_real = Os.shape[0]

	bs_iter = (np.nanmean(Os[np.random.randint(n_real,size=n_real)],axis=0) for i in range(n_bs))
	diff_iter = ((bs-avg)**2 for bs in bs_iter)
	err = np.sqrt(sum(diff_iter)/n_bs)

	return avg,err

def bootstrap_fluc(Os,n_bs=100):
	Os = np.asarray(Os)
	avg = np.nanstd(Os,axis=0)
	n_real = Os.shape[0]

	bs_iter = (np.nanstd(Os[np.random.randint(n_real,size=n_real)],axis=0) for i in range(n_bs))
	diff_iter = ((bs-avg)**2 for bs in bs_iter)
	err = np.sqrt(sum(diff_iter)/n_bs)

	return avg,err


def get_filedict(filename):
	filename = os.path.split(filename)[-1] # remove path
	filename = ".".join(filename.split(".")[:-1]) # rm extension
	filelist = filename.split("_")

	filedict = {}
	for i,item in enumerate(filelist):
		if item.replace(".","").isdigit():
			filedict[filelist[i-1]] = eval(item)

	return filedict





filelist = glob.glob(sys.argv[1])
filelist.sort()

data_dict = {}

L_list = set([])
Nc_list = set([])
T_list = set([])

print filelist

if filelist:
	for filename in filelist:
		print filename 	
		filedict = get_filedict(filename)

		L = filedict["L"]
		T = filedict["T"]
		Nc = filedict["Nc"]

		L_list.add(L)
		T_list.add(T)
		Nc_list.add(Nc)


		data = np.loadtxt(filename)
		if data.size == 0: 
			continue 
		Q,M2 = data[:,0],data[:,1]

		Q_avg,dQ_avg = bootstrap_mean(Q)
		M2_avg,dM2_avg = bootstrap_mean(M2)
		row = [Q_avg,dQ_avg,M2_avg,dM2_avg]

		data_dict[(L,Nc,T)] = np.array(row)


	L_list = np.fromiter(L_list,dtype=np.int)
	T_list = np.fromiter(T_list,dtype=np.float)
	Nc_list = np.fromiter(Nc_list,dtype=np.int)

	L_list.sort()
	T_list.sort()
	Nc_list.sort()

	shape = L_list.shape+Nc_list.shape+T_list.shape+(4,)
	data = np.full(shape,np.nan,dtype=np.float)
	for L,Nc,T in product(L_list,Nc_list,T_list):
		if (L,Nc,T) in data_dict:
			i = np.searchsorted(L_list,L)
			j = np.searchsorted(Nc_list,Nc)
			k = np.searchsorted(T_list,T)
			data[i,j,k,:] = data_dict[(L,Nc,T)]


	np.savez_compressed("runs.npz",data=data,L_list=L_list,Nc_list=Nc_list,T_list=T_list)

