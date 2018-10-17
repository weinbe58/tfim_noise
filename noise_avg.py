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
list_dict = {}

print filelist

if filelist:
	for filename in filelist:
		print filename 	
		filedict = get_filedict(filename)

		for key,value in filedict.items():
			if key in list_dict:
				list_dict[key].add(value)
			else:
				list_dict[key] = set([value])


		data = np.loadtxt(filename)
		if data.size == 0: 
			continue 
		Q,M2 = data[:,0],data[:,1]

		Q_avg,dQ_avg = bootstrap_mean(Q)
		M2_avg,dM2_avg = bootstrap_mean(M2)
		row = [Q_avg,dQ_avg,M2_avg,dM2_avg]

		key = tuple(filedict.items())
		data_dict[key] = np.array(row)

	keys = list_dict.keys()
	keys.sort(key = lambda x:(-len(list_dict[x]),x))
	print keys
	shape = ()
	for key in keys:
		np_list = np.fromiter(list_dict[key],dtype=np.float)
		np_list.sort()
		list_dict[key] = np_list
		shape = shape + np_list.shape


	shape = shape + (4,)
	data = np.full(shape,np.nan,dtype=np.float)

	for key,d in data_dict.items():
		key_dict = dict(key)
		index = []
		for file_key in keys:
			i = np.searchsorted(list_dict[file_key],key_dict[file_key])
			index.append((i,))
		index.append(Ellipsis)
		data[tuple(index)] = d

	np.savez_compressed(sys.argv[2],data=data,**list_dict)

