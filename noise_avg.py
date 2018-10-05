import numpy as np
import glob,sys,os 
import matplotlib.pyplot as plt


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


# filelist = glob.glob("/project/diamond/mbl_1d_data/run_L_*_w_*.dat")
filelist = glob.glob("/projectnb/qspin/weinbe58/project_temp_data/mbl_1d_data/run_L_*_w_*.dat")
filelist.sort()

file_dict = {}

if filelist:
	for filename in filelist:
		print filename 	
		filelist = os.path.split(filename)[-1].replace(".dat","").split("_")

		i = filelist.index("L")
		N = int(filelist[i+1])
		i = filelist.index("w")
		w = float(filelist[i+1])
		if os.path.isfile("ramp_L_{}.out".format(N)):
			continue

		if N not in file_dict:
			file_dict[N] = {}

		file_dict[N][w] = filename


	for N in file_dict.keys():
		print "ramp_L_{}.out".format(N)	
		if os.path.isfile("ramp_L_{}.out".format(N)):
			continue

		ws = np.asarray(file_dict[N].keys())

		arg = np.argsort(ws)
		ws = ws[arg].reshape((-1,1))

		y = []
		for w in ws.ravel():
			print file_dict[N][w]
			data = np.loadtxt(file_dict[N][w])
			continue 
			MLS = data[:,0]
			shape = MLS.shape+(2,-1)
			other_data = data[:,1:].reshape(shape)
			F = np.nanmean(other_data[:,0,:],axis=1)
			S = np.nanmean(other_data[:,1,:],axis=1)
		
			S,dS = bootstrap_mean(S)
			F_e,dF_e = bootstrap_mean(F)
			F2_e,dF2_e = bootstrap_mean(F**2)
			MLS,dMLS = bootstrap_mean(MLS)

			y.append([F_e,dF_e,F2_e,dF2_e,S,dS,MLS,dMLS])


		continue 
		y = np.asarray(y)
		data = np.hstack((ws,y))

		np.savetxt("ramp_L_{}.out".format(N),data,fmt="%30.15e")
		


