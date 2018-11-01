import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import glob,sys,os


def color_list(i):
	col=['red','green','blue','fuchsia','yellow','orange','purple','gray','aqua','lime','maroon','teal','navy','black' , 'olive','silver','lawngreen','dodgerblue','MediumSpringGreen','DarkOrange']     
	j=i % len(col)
	return col[j]

def fill_marker(i):
	m = len(Line2D.filled_markers)
	j=i % m
	return Line2D.filled_markers[j]


def plot(datadict,figname,ncol,xlabel,ylabel,error=True,keys=None,logx=True,logy=False,legend_opts={},xscale=0.0,yscale=0.0,yshift=None,xshift=None,xlim=None,ylim=None,legend=True,func=None):
	plt.clf()
	if xshift is None:
		xshift = lambda v,L:0.0
	if yshift is None:
		yshift = lambda v,L:0.0

	if keys is None:
		keys = datadict.keys()
		keys.sort(key=lambda x:x[0])

	for i,key in enumerate(keys):
		L,label = key
		data = datadict[key]
		v = 1.0/data[:,0]
		if error:
			plt.errorbar((v-xshift(v,L))*L**xscale,(data[:,ncol]-yshift(v,L))*L**yscale,data[:,ncol+1]*L**yscale,marker=fill_marker(i),
				color=color_list(i),label=label)
		else:
			plt.plot((v-xshift(v,L))*L**xscale,(data[:,ncol]-yshift(v,L))*L**yscale,marker=fill_marker(i),
				color=color_list(i),label=label)


	if logx:
		plt.xscale("log", nonposx='clip')
	if logy:
		plt.yscale("log", nonposy='clip')

	xmin,xmax = plt.gca().get_xlim()
	ymin,ymax = plt.gca().get_ylim()

	if func is not None:
		xx = np.linspace(xmin,xmax,10001)
		plt.plot(xx,func(xx),linestyle=":",color=color_list(len(keys)),marker="")

	plt.xlim(xmin,xmax)
	plt.ylim(ymin,ymax)

	if legend:
		plt.legend(**legend_opts)
	plt.xlabel(xlabel,fontsize=16)
	plt.ylabel(ylabel,fontsize=16)
	if xlim is not None:
		plt.xlim(xlim)
	if ylim is not None:
		plt.ylim(ylim)
	plt.savefig(figname,dpi=1000)


datafile = sys.argv[1]
runs = np.load(datafile)
keys = runs.keys()
keys.sort()
print keys



data = runs["data"]
L_list = runs["L"]
T_list = runs["T"]
print runs["Nb"]
datadict = {}

print data.shape
for i,L in enumerate(L_list):
	if L < 10:
		continue

	key = (L,"$L={}$".format(int(L)))

	d = np.hstack((np.atleast_2d(T_list).T,data[i,i,:,1,0]))
	# d = np.hstack((np.atleast_2d(T_list).T,data[i,-1,:,0,0]))
	# d = np.hstack((np.atleast_2d(T_list).T,data[i,:,-1,0,:]))


	datadict[key] = d



func_e = lambda x:np.sqrt(x)
func_m2 = lambda x:1/x

exp = [0.0,1.0,2.0,2*(1+1.0/3.0),3.0]
for e in exp:
	error = False
	logy = False
	logx = True
	plot(datadict,"{}_m2_{}.png".format(".".join(os.path.split(datafile)[-1].split(".")[:-1]),e),2,
		"$vL^{{{}}}$".format(e),"$m^2(v,L)$",error=error,xscale=e,logx=logx,logy=logy)

	# logy = True
	plot(datadict,"{}_e_{}.png".format(".".join(os.path.split(datafile)[-1].split(".")[:-1]),e),1,
		"$vL^{{{}}}$".format(e),"$Q(v,L)/L$",error=error,xscale=e,logx=logx,logy=logy,func=func_e)
