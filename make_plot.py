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
	xscale=0.0,yscale=0.0,yshift=None,xshift=None,xlim=None,ylim=None,legend=False,error=True):



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
		if error:
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



datafile = sys.argv[1]
runs = np.load(datafile)
keys = runs.keys()
keys.sort()
print keys



data = runs["data"]
L_list = runs["L"]
T_list = runs["T"]
datadict = {}

print data.shape
for i,L in enumerate(L_list):
	if L < 10: continue

	key = (L,"$L={}$".format(int(L)))
	mask = T_list > 0

	d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,i,mask,0,0]))
	# d = np.hstack((np.atleast_2d(1/T_list).T,data[i,mask,0]))
	# d = np.hstack((np.atleast_2d(1.0/T_list[mask]).T,data[i,mask,-1,0,:]))


	datadict[key] = d

# Two subplots, the axes array is 1-d
width = 3.39
height =  1.5*width

f, (ax1,ax2) = plt.subplots(2,figsize=(width,height))

options = dict(logy=False,logx=True,error=False,xlabel="$vL^2$",
	ylabel="$Q(v,L)$",xscale=2,yscale=0.0)

plot(ax1,datadict,1,legend=True,legend_opts=dict(ncol=1,fontsize=6),**options)


options["ylabel"] = "$m^2(v,L)$"
options["yscale"] = 0.0
plot(ax2,datadict,2,**options)
f.text(0.025,0.95,"$(a)$",fontsize=12)
f.text(0.025,0.465,"$(b)$",fontsize=12)
plt.tight_layout()

f.savefig(os.path.join(".","model_scale_2.pdf"),bbox_inches="tight")
plt.clf()

# Two subplots, the axes array is 1-d
width = 3.39
height =  1.5*width

f, (ax1,ax2) = plt.subplots(2,figsize=(width,height))

options = dict(logy=True,logx=True,error=False,xlabel="$vL^3$",
	ylabel="$Q(v,L)$",xscale=1,yscale=0.0)

plot(ax1,datadict,1,legend=True,legend_opts=dict(ncol=1,fontsize=6),**options)

xmin,xmax = ax1.get_xlim()
ymin,ymax = ax1.get_ylim()


x = np.linspace(xmin,xmax,1000)
ax1.plot(x,10*np.sqrt(x),label="linear")

options["ylabel"] = "$m^2(v,L)$"
options["yscale"] = 0.0
plot(ax2,datadict,2,**options)
f.text(0.025,0.95,"$(a)$",fontsize=12)
f.text(0.025,0.465,"$(b)$",fontsize=12)
plt.tight_layout()

f.savefig(os.path.join(".","model_scale_3.pdf"),bbox_inches="tight")