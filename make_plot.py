import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import glob,sys,os


def marker_style(i):
	color = ['red','green','blue','fuchsia','yellow','orange','aqua','lime','teal']
	n_color = len(color)
	n_marker = len(Line2D.filled_markers)


	markers = Line2D.filled_markers
	fillstyle = ["full","none","top","bottom","left","right"]

	j = i % n_color
	jj = (i // n_color) % len(fillstyle)

	return dict(color=color[j],marker=markers[j],fillstyle=fillstyle[jj])


def plot(ax,filedict,ncol,ecol=None,xlabel=None,ylabel=None,keys=None,logx=True,logy=False,legend_opts={},
	xscale=0.0,yscale=0.0,yshift=None,xshift=None,xlim=None,ylim=None,legend=False):



	if xshift is None:
		xshift = lambda v,L:0.0
	if yshift is None:
		yshift = lambda v,L:0.0

	if keys is None:
		keys = filedict.keys()
		keys.sort(key=lambda x:x[0])

	for i,key in enumerate(keys):
		L,label = key
		data =filedict[key]
		v = data[:,0]
		y = (data[:,ncol]-yshift(v,L))*L**yscale
		x = (v-xshift(v,L))*L**xscale
		if ecol is not None:
			err = data[:,ecol]*L**yscale
			ax.errorbar(x,y,err,label=label,markersize=3,linewidth=1,**marker_style(i))
		else:
			ax.plot(x,y,label=label,markersize=3,linewidth=1,**marker_style(i))

	if logx:
		ax.set_xscale("log", nonposx='clip')
		ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
	if logy:
		ax.set_yscale("log", nonposy='clip')
		ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))

	if xlabel is not None:
		ax.set_xlabel(xlabel,fontsize=10)
	if ylabel is not None:
		ax.set_ylabel(ylabel,fontsize=10)
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)

	if legend:
		ax.legend(**legend_opts)


	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(7) 

	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(7) 

	ax.tick_params(axis="both",which="both",direction="in")

def spin_bath_1d_2():
	datafile = "data/spin_bath_2.npz"
	runs = np.load(datafile)
	keys = runs.keys()
	keys.sort()
	print keys

	data = runs["data"]

	if "L" in runs:
		L_list = runs["L"]
	elif "size" in runs:
		L_list = runs["size"]

	T_list = runs["T"]
	datadict = {}

	print data.shape
	for i,L in enumerate(L_list):
		try:
			n,m = L

			if n == 0:
				key = (m,"$L={}$".format(int(m)))
			else:
				key = (np.sqrt(n**2+m**2),r"$L=\sqrt{{{}}}$".format(n**2+m**2))

			mask = T_list > 0
			d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,mask,0,0,i]))
			yscale=-1.0
		except TypeError:
			if L < 10: continue
			key = (L,"$L={}$".format(int(L)))


			mask = T_list >= L**2/10.0

			d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,i,mask,0,0]))
			# d = np.hstack((np.atleast_2d(1/T_list).T,data[i,mask,0]))
			# d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,mask,-1,0,:]))
			yscale=0.0


		datadict[key] = d

	# Two subplots, the axes array is 1-d
	width = 3.39
	height =  1.5*width

	f, (ax1,ax2) = plt.subplots(2,figsize=(width,height))

	options = dict(logy=False,logx=True,xlabel="$vL^2$",
		ylabel="$Q(v,L)$",xscale=2,yscale=yscale)

	plot(ax1,datadict,1,legend=True,legend_opts=dict(ncol=1,fontsize=6),**options)


	options["ylabel"] = "$m^2(v,L)$"
	options["yscale"] = 0.0
	options["logy"] = True
	plot(ax2,datadict,2,**options)
	f.text(0.025,0.95,"$(a)$",fontsize=12)
	f.text(0.025,0.465,"$(b)$",fontsize=12)
	plt.tight_layout()

	f.savefig(os.path.join(".","model_xy_scale_2.pdf"),bbox_inches="tight")
	plt.clf()

	# Two subplots, the axes array is 1-d
	width = 3.39
	height =  1.5*width
	xscale = 3.0

	f, (ax1,ax2) = plt.subplots(2,figsize=(width,height))

	options = dict(logy=False,logx=True,xlabel="$vL^3$",
		ylabel="$Q(v,L)$",xscale=xscale,yscale=yscale)

	plot(ax1,datadict,1,legend=True,legend_opts=dict(ncol=1,fontsize=6),**options)



	options["ylabel"] = "$m^2(v,L)$"
	options["yscale"] = 0.0
	options["logy"] = True
	plot(ax2,datadict,2,**options)


	xmin,xmax = ax2.get_xlim()
	ymin,ymax = ax2.get_ylim()


	# x = np.linspace(xmin,xmax,1000)
	# ax2.plot(x,x**(-1),label="linear")
	# ax2.set_ylim(ymin,ymax)



	f.text(0.025,0.95,"$(a)$",fontsize=12)
	f.text(0.025,0.465,"$(b)$",fontsize=12)
	plt.tight_layout()

	f.savefig(os.path.join(".","model_xy_scale_3.pdf"),bbox_inches="tight")

def spin_bath_1d_5():
	datafile = "data/spin_bath_5.npz"
	runs = np.load(datafile)
	keys = runs.keys()
	keys.sort()
	print keys

	data = runs["data"]

	if "L" in runs:
		L_list = runs["L"]
	elif "size" in runs:
		L_list = runs["size"]

	T_list = runs["T"]
	datadict = {}

	print data.shape
	for i,L in enumerate(L_list):
		try:
			n,m = L

			if n == 0:
				key = (m,"$L={}$".format(int(m)))
			else:
				key = (np.sqrt(n**2+m**2),r"$L=\sqrt{{{}}}$".format(n**2+m**2))

			mask = T_list > 0
			d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,mask,0,0,i]))
			yscale=-1.0
		except TypeError:
			if L < 10: continue
			key = (L,"$L={}$".format(int(L)))


			mask = T_list < 2*L

			d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,i,mask,0,0]))
			# d = np.hstack((np.atleast_2d(1/T_list).T,data[i,mask,0]))
			# d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,mask,-1,0,:]))
			yscale=0.0


		datadict[key] = d

	# Two subplots, the axes array is 1-d
	width = 3.39
	height =  1.5*width

	f, (ax1,ax2) = plt.subplots(2,figsize=(width,height))

	options = dict(logy=False,logx=True,xlabel="$vL^2$",
		ylabel="$Q(v,L)$",xscale=2,yscale=yscale)

	plot(ax1,datadict,1,legend=True,legend_opts=dict(ncol=1,fontsize=6),**options)


	options["ylabel"] = "$m^2(v,L)$"
	options["yscale"] = 0.0
	options["logy"] = True
	plot(ax2,datadict,2,**options)
	f.text(0.025,0.95,"$(a)$",fontsize=12)
	f.text(0.025,0.465,"$(b)$",fontsize=12)
	plt.tight_layout()

	f.savefig(os.path.join(".","model_SU2_1d_scale_2.pdf"),bbox_inches="tight")
	plt.clf()

	# Two subplots, the axes array is 1-d
	width = 3.39
	height =  1.5*width
	xscale = 1.0

	f, (ax1,ax2) = plt.subplots(2,figsize=(width,height))

	options = dict(logy=False,logx=True,xlabel="$vL$",
		ylabel="$Q(v,L)$",xscale=xscale,yscale=yscale)

	plot(ax1,datadict,1,legend=True,legend_opts=dict(ncol=1,fontsize=6),**options)



	options["ylabel"] = "$m^2(v,L)$"
	options["yscale"] = 0.0
	options["logy"] = True
	plot(ax2,datadict,2,**options)


	xmin,xmax = ax2.get_xlim()
	ymin,ymax = ax2.get_ylim()


	# x = np.linspace(xmin,xmax,1000)
	# ax2.plot(x,x**(-1),label="linear")
	# ax2.set_ylim(ymin,ymax)



	f.text(0.025,0.95,"$(a)$",fontsize=12)
	f.text(0.025,0.465,"$(b)$",fontsize=12)
	plt.tight_layout()

	f.savefig(os.path.join(".","model_SU2_1d_scale_1.pdf"),bbox_inches="tight")

def spin_bath_2d_1():
	datafile = "data/2d_spin_bath_1.npz"
	runs = np.load(datafile)
	keys = runs.keys()
	keys.sort()
	print keys

	data = runs["data"]

	if "L" in runs:
		L_list = runs["L"]
	elif "size" in runs:
		L_list = runs["size"]

	T_list = runs["T"]
	datadict = {}

	print data.shape
	for i,L in enumerate(L_list):
		try:
			n,m = L

			if n == 0:
				key = (m,"$L={}$".format(int(m)))
			else:
				key = (np.sqrt(n**2+m**2),r"$L=\sqrt{{{}}}$".format(n**2+m**2))


			mask = T_list < 10*np.sqrt(n**2+m**2)
			d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,mask,0,0,i]))
			yscale=-1.0
		except TypeError:
			if L < 10: continue
			key = (L,"$L={}$".format(int(L)))


			mask = T_list > 0

			d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,i,mask,0,0]))
			# d = np.hstack((np.atleast_2d(1/T_list).T,data[i,mask,0]))
			# d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,mask,-1,0,:]))
			yscale=0.0


		datadict[key] = d

	# Two subplots, the axes array is 1-d
	width = 3.39
	height =  1.5*width

	f, (ax1,ax2) = plt.subplots(2,figsize=(width,height))

	options = dict(logy=False,logx=True,xlabel="$vL^2$",
		ylabel="$Q(v,L)$",xscale=2,yscale=yscale)

	plot(ax1,datadict,1,legend=True,legend_opts=dict(ncol=1,fontsize=6),**options)


	options["ylabel"] = "$m^2(v,L)$"
	options["yscale"] = 0.0
	options["logy"] = True
	plot(ax2,datadict,2,**options)
	f.text(0.025,0.95,"$(a)$",fontsize=12)
	f.text(0.025,0.465,"$(b)$",fontsize=12)
	plt.tight_layout()

	f.savefig(os.path.join(".","model_SU2_2d_scale_2.pdf"),bbox_inches="tight")
	plt.clf()

	# Two subplots, the axes array is 1-d
	width = 3.39
	height =  1.5*width
	xscale = 1.0

	f, (ax1,ax2) = plt.subplots(2,figsize=(width,height))

	options = dict(logy=False,logx=True,xlabel="$vL$",
		ylabel="$Q(v,L)$",xscale=xscale,yscale=yscale)

	plot(ax1,datadict,1,legend=True,legend_opts=dict(ncol=1,fontsize=6),**options)



	options["ylabel"] = "$m^2(v,L)$"
	options["yscale"] = 0.0
	options["logy"] = True
	plot(ax2,datadict,2,**options)


	xmin,xmax = ax2.get_xlim()
	ymin,ymax = ax2.get_ylim()


	# x = np.linspace(xmin,xmax,1000)
	# ax2.plot(x,x**(-1),label="linear")
	# ax2.set_ylim(ymin,ymax)



	f.text(0.025,0.95,"$(a)$",fontsize=12)
	f.text(0.025,0.465,"$(b)$",fontsize=12)
	plt.tight_layout()

	f.savefig(os.path.join(".","model_SU2_2d_scale_1.pdf"),bbox_inches="tight")

def tfim_1d_snake():
	datafile = "data/snake_2.npz"
	runs = np.load(datafile)
	keys = runs.keys()
	keys.sort()
	print keys

	data = runs["data"]

	if "L" in runs:
		L_list = runs["L"]
	elif "size" in runs:
		L_list = runs["size"]

	T_list = runs["T"]
	datadict = {}

	print data.shape
	for i,L in enumerate(L_list):
		key = (L,"$L={}$".format(int(L)))
		d = np.hstack((np.atleast_2d(1.0/T_list).T,data[0,i,0,:]))
		yscale=0.0


		datadict[key] = d

	# Two subplots, the axes array is 1-d
	width = 3.39
	height =  1.5*width

	f, (ax1,ax2) = plt.subplots(2,figsize=(width,height))

	options = dict(logy=False,logx=True,xlabel="$vL^2$",
		ylabel="$Q(v,L)$",xscale=2)

	plot(ax1,datadict,1,ecol=2,legend=True,legend_opts=dict(ncol=1,fontsize=6),**options)


	options["ylabel"] = "$m^2(v,L)$"
	options["logy"] = True
	plot(ax2,datadict,3,ecol=4,**options)
	f.text(0.025,0.95,"$(a)$",fontsize=12)
	f.text(0.025,0.465,"$(b)$",fontsize=12)
	plt.tight_layout()

	f.savefig(os.path.join(".","model_snake_scale_2.pdf"),bbox_inches="tight")
	plt.clf()




spin_bath_2d_1()
spin_bath_1d_5()
spin_bath_1d_2()
tfim_1d_snake()