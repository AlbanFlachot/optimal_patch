from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def DEFINE_PLT_RC(type = 1):
    '''
    Function taht sets the rc parameters of matplot lib.e.g in an article
    e.g. in an article, whether it will be an image with columnwidth, or half of it, full page etc.. 
    INPUT:
        type: 1 = full page size; 1/2 = half of page size; 1/4 = quarter of page size; 1/3 = third of page size
    '''
    
    plt.rc('figure', figsize = (7,5))     # fig size bigger
    if type == 0:
        #plt.rc('font', weight='bold')    # bold fonts are easier to see
        plt.rc('xtick', labelsize=7.5)
        plt.rc('ytick', labelsize=7.5)# tick labels bigger
        plt.rc('axes', labelsize=8.5)     # tick labels bigger
        plt.rc('lines', lw=2.5) # thicker black lines
        plt.rc('text', fontsize=8.5) # thicker black lines
    elif type == 1:
        #plt.rc('font', weight='bold')    # bold fonts are easier to see
        plt.rc('xtick', labelsize=15)     # tick labels bigger
        plt.rc('ytick', labelsize=15)     # tick labels bigger
        plt.rc('axes', labelsize=17)     # tick labels bigger
        plt.rc('lines', lw=5) # thicker black lines
        plt.rc('text', fontsize=17) # thicker black lines
    elif type == 0.5:
        #plt.rc('font', weight='bold')    # bold fonts are easier to see
        plt.rc('xtick', labelsize=30)     # tick labels bigger
        plt.rc('ytick', labelsize=30)     # tick labels bigger
        plt.rc('axes', labelsize=35)     # tick labels bigger
        plt.rc('lines', lw=10) # thicker black lines
        plt.rc('text', fontsize=35) # thicker black lines
    elif type == 0.33:
        #plt.rc('font', weight='bold')    # bold fonts are easier to see
        plt.rc('xtick', labelsize=24)     # tick labels bigger
        plt.rc('ytick', labelsize=24)     # tick labels bigger
        plt.rc('axes', labelsize=28)     # tick labels bigger
        plt.rc('lines', lw=7.5) # thicker black lines
        plt.rc('text', fontsize=28) # thicker black lines
        
def respo(X,t):
	return np.sum((np.nanmax(X[:,1:],axis = -1)>t)*1)/(np.nanmean(X[:,1:],axis = -1).size)
	#return np.sum((np.nanmax(X[:,:],axis = 0)>t)*1)/(np.nanmean(X[:,:],axis = 0).size)  #For chromatic contrast responsivity

def respo2(X,t):
	#return np.sum((np.nanmax(X[:,1:],axis = -1)>t)*1)/(np.nanmean(X[:,1:],axis = -1).size)
	return np.sum((np.nanmax(X[:,:],axis = 0)>t)*1)/(np.nanmean(X[:,:],axis = 0).size)  #For chromatic contrast responsivity

def RESPO(RESP,t1,t2,t3, resp_type = 1):
	respt1 = np.zeros(len(RESP))
	respt2 = np.zeros(len(RESP))
	respt3 = np.zeros(len(RESP))
	Mean_resp = np.zeros(len(RESP))
	Std_resp = np.zeros(len(RESP))
	if resp_type == 1:
        	for l in range(0,len(RESP)):
        		respt1[l] = respo(RESP[l],t1)
        		respt2[l] = respo(RESP[l],t2)
        		respt3[l] = respo(RESP[l],t3)
        		Mean_resp[l] = np.nanmean(RESP[l])
        		Std_resp[l] = np.nanstd(RESP[l])
	else:
        	for l in range(0,len(RESP)):
        		respt1[l] = respo2(RESP[l],t1)
        		respt2[l] = respo2(RESP[l],t2)
        		respt3[l] = respo2(RESP[l],t3)
        		Mean_resp[l] = np.nanmean(RESP[l])
        		Std_resp[l] = np.nanstd(RESP[l])
	return respt1,respt2,respt3,Mean_resp,Std_resp

def SENS(RESP,t1,t2,t3):
	respt1 = np.zeros(len(RESP))
	respt2 = np.zeros(len(RESP))
	respt3 = np.zeros(len(RESP))
	Mean_resp = np.zeros(len(RESP))
	Std_resp = np.zeros(len(RESP))
	for l in range(0,len(RESP)):
		respt1[l] = respo(RESP[l],t1)
		respt2[l] = respo(RESP[l],t2)
		respt3[l] = respo(RESP[l],t3)
		Mean_resp[l] = np.mean(RESP[l])
		Std_resp[l] = np.std(RESP[l])
	return respt1,respt2,respt3,Mean_resp,Std_resp

def PLOT_FIGURE(measure,respt1,respt2,respt3,Mean_resp):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(np.arange(1,len(respt1)+1),respt1,'k:',linewidth = 4,label = measure +' > 0.25')
	ax.plot(np.arange(1,len(respt1)+1),respt2,'k--',linewidth = 4,label = measure +'  > 0.50')
	ax.plot(np.arange(1,len(respt1)+1),respt3,'k-',linewidth = 4,label = measure +'  > 0.75')
	#ax.errorbar(np.arange(1,6),Mean_resp,yerr =Std_resp,marker = 'o',color = 'r',markersize = 4,linewidth = 0,ecolor = 'r',elinewidth = 1,capsize = 2)
	ax.plot(np.arange(1,len(respt1)+1),Mean_resp,'ro',ms = 8)
	#ax.plot(np.arange(1,8),Comp,'o',ms = 12,label = '1-<r$^2$>$^2$')
	plt.setp(ax.get_xticklabels(), fontsize=20)
	plt.setp(ax.get_yticklabels(), fontsize=20)
	plt.xlabel('Layer',fontsize = 25)
	plt.ylabel('Proportion of kernels',fontsize = 25)
	#plt.ylim(-0.05,0.55)
	plt.ylim(0,1)
	plt.xlim(0.5,len(respt1)+0.5)
	plt.legend(loc='upper center', fontsize = 20)
	fig.tight_layout()
	fig.show()
	#fig.savefig('tuning_curves/BVLC: Responsivity (normalized)')

	plt.close(fig)

def PLOT_FIGURE_GRADUATE_DISTRIB(DIS_R,measure):

	rect_ax4 = [0.1, 0.1, 0.6, 0.85]
	fig = plt.figure(figsize = (7,5))
	ax4 = plt.axes(rect_ax4)
	ax4.plot(np.arange(1,len(DIS_R)+1),DIS_R,'k',linewidth = 1)
	ax4.fill_between(np.arange(1,len(DIS_R)+1),DIS_R[:,0],100,color = [1,1,1],label = measure)
	ax4.plot(np.arange(1,len(DIS_R)+1),DIS_R[:,1],'k',linewidth = 1)
	ax4.fill_between(np.arange(1,len(DIS_R)+1),DIS_R[:,1],DIS_R[:,0],color = [0.8,0.8,0.8],label = '> 0')
	ax4.plot(np.arange(1,len(DIS_R)+1),DIS_R[:,2],'k-',linewidth = 1)
	ax4.fill_between(np.arange(1,len(DIS_R)+1),DIS_R[:,2],DIS_R[:,1],color = [0.70,0.70,0.70],label = '> 1/8')
	ax4.plot(np.arange(1,len(DIS_R)+1),DIS_R[:,3],'k-',linewidth = 1)
	ax4.fill_between(np.arange(1,len(DIS_R)+1),DIS_R[:,3],DIS_R[:,2],color = [0.60,0.60,0.60],label = '> 1/4')
	ax4.plot(np.arange(1,len(DIS_R)+1),DIS_R[:,4],'k-',linewidth = 1)
	ax4.fill_between(np.arange(1,len(DIS_R)+1),DIS_R[:,4],DIS_R[:,3],color = [0.50,0.50,0.50],label = '> 3/8')
	ax4.plot(np.arange(1,len(DIS_R)+1),DIS_R[:,5],'k-',linewidth = 1)
	ax4.fill_between(np.arange(1,len(DIS_R)+1),DIS_R[:,5],DIS_R[:,4],color = [0.4,0.4,0.40],label = '> 1/2')
	ax4.plot(np.arange(1,len(DIS_R)+1),DIS_R[:,6],'k-',linewidth = 1)
	ax4.fill_between(np.arange(1,len(DIS_R)+1),DIS_R[:,6],DIS_R[:,5],color = [0.3,0.3,0.30],label = '> 5/8')
	ax4.plot(np.arange(1,len(DIS_R)+1),DIS_R[:,7],'k-',linewidth = 1)
	ax4.fill_between(np.arange(1,len(DIS_R)+1),DIS_R[:,7],DIS_R[:,6],color = [0.2,0.2,0.2],label = '> 3/4')
	ax4.fill_between(np.arange(1,len(DIS_R)+1),0,DIS_R[:,7],color = [0.1,0.1,0.1],label = '> 7/8')
	#ax.plot(numLay,Mean_selec_HSENS1,'ro',ms = 8)
	#ax.plot(np.arange(1,8),Comp,'o',ms = 12,label = '1-<r$^2$>$^2$')
	plt.xlabel('Layer')
	plt.ylabel('Percentage of units')

	ax4.set_xticks(np.arange(1,len(DIS_R)+1,len(DIS_R)//5))
	ax4.legend(bbox_to_anchor=(1.01, 0, 0.42, 0.95), loc=1,
           ncol=1, mode="expand", borderaxespad=0.,fontsize=20)
	ax4.set_xlim([1,len(DIS_R)])
	#fig.tight_layout()
	fig.show()
	#print DIS_R[:,1]


def plot_fig_summary(respt2,Mean_resp):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(np.arange(1,len(respt2)+1),respt2,'k',linewidth = 6,label='proportion of\ncolor responsive\nkernels')
	ax.plot(np.arange(1,len(respt2)+1),Mean_resp,color = 'r',linewidth = 6,label = 'mean color\nresponsivity')
	plt.xlabel('Layer')
	#plt.ylabel('Proportion of kernels',fontsize = 25)
	#plt.ylim(-0.05,0.55)
	plt.ylim(0,1)
	plt.xlim(0.5,len(respt2)+0.5)
	#plt.legend(loc='best', fontsize = 20)
	#fig.tight_layout()
	plt.show()
	#fig.savefig('tuning_curves/BVLC: Responsivity (normalized)')
	plt.close(fig)

def plot_horizontal_histo(DATA, Bins, y_label, title_fig, name_net):
    import math
    step_plot = math.ceil(float(len(DATA))/5)
    
    lay2dis = range(0,len(DATA)+1,int(step_plot)) # defines which layer to display given that we want to display only 5 and the nets have diferent nb of layers.
    lay2dis[-1] = len(DATA)-1
    
    # definitions for the axes
    left, width = 0.12, 0.15
    bottom, height = 0.2, 0.72
    #bottom_h = left_h = left + width + 0.01
    
    rect_histy1 = [left, bottom, width, height]
    
    #rect_histx = [left, bottom_h, width, 0.2]
    rect_histy2 = [left +width + 0.03, bottom, width, height]
    rect_histy3 = [left +width + 0.03+width + 0.03, bottom, width, height]
    rect_histy4 = [left +width + 0.03+width + 0.03+width + 0.03, bottom, width, height]
    rect_histy5 = [left +width + 0.03+width + 0.03+width + 0.03+width + 0.03, bottom, width, height]
    
    fig = plt.figure()

    
    ax1 = plt.axes(rect_histy1)
    
    #hist_hue = np.histogram( np.concatenate((ARG_HSENS1[i][M_HSENS1[i]>t2],ARG_HSENS2[i][M_HSENS2[i]>t2])),bins = Bins )
    hist_hue = np.histogram( DATA[0][~np.isnan(DATA[0])],bins = Bins )
    Y = (hist_hue[0])/DATA[0][~np.isnan(DATA[0])].size
    ax1.barh(hist_hue[1][:-1],Y, height = Bins[1] - Bins[0],color = 'k')
    
    ax2 = plt.axes(rect_histy2)
    hist_hue = np.histogram( DATA[lay2dis[1]][~np.isnan(DATA[lay2dis[1]])],bins = Bins )
    Y = (hist_hue[0])/DATA[lay2dis[1]][~np.isnan(DATA[lay2dis[1]])].size
    ax2.barh(hist_hue[1][:-1] , Y, height = Bins[1] - Bins[0],color = 'k')
    
    ax3 = plt.axes(rect_histy3)
    hist_hue = np.histogram( DATA[lay2dis[2]][~np.isnan(DATA[lay2dis[2]])],bins = Bins )
    Y = (hist_hue[0])/DATA[lay2dis[2]][~np.isnan(DATA[lay2dis[2]])].size
    ax3.barh(hist_hue[1][:-1], Y, height = Bins[1] - Bins[0],color = 'k')
    
    ax4 = plt.axes(rect_histy4)
    hist_hue = np.histogram( DATA[lay2dis[3]][~np.isnan(DATA[lay2dis[3]])],bins = Bins )
    Y = (hist_hue[0])/DATA[lay2dis[3]][~np.isnan(DATA[lay2dis[3]])].size
    ax4.barh(hist_hue[1][:-1], Y, height = Bins[1] - Bins[0],color = 'k')
    
    ax5 = plt.axes(rect_histy5)
    hist_hue = np.histogram( DATA[lay2dis[4]][~np.isnan(DATA[lay2dis[4]])],bins = Bins )
    Y = (hist_hue[0])/DATA[lay2dis[4]][~np.isnan(DATA[lay2dis[4]])].size
    ax5.barh(hist_hue[1][:-1], Y, height = Bins[1] - Bins[0],color = 'k')
    
    #plt.setp(ax1.get_xticklabels(),fontsize=15)
    #plt.setp(ax1.get_yticklabels(),fontsize=15)
    
    #ax1.set_yticks((Bins))
    ax1.set_xticks([])
    ax1.set_ylabel(y_label)
    ax1.set_xlabel('%s' %str(lay2dis[0]))
    ax1.set_ylim([0,Bins[-1]])
    
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlabel('%s' %str(lay2dis[1]))
    ax2.set_ylim([0,Bins[-1]])
    
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_xlabel('%s' %str(lay2dis[2]))
    ax3.set_ylim([0,Bins[-1]])
    
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_xlabel('%s' %str(lay2dis[3]))
    ax4.set_ylim([0,Bins[-1]])
    
    ax5.set_yticks([])
    ax5.set_xticks([])
    ax5.set_xlabel('%s' %str(lay2dis[4]))
    ax5.set_ylim([0,Bins[-1]])
    
    fig.text(0.54, 0.02, 'Layer', ha='center')
    
    #fig.tight_layout()
    plt.show()
    fig.savefig('../figures/' + title_fig + '_' + name_net,dpi = 300)
    plt.close()
    
def plot_vertical_histo(DATA, Bins, y_label, title_fig, name_net, mean = False):
    import math
    step_plot = math.ceil(float(len(DATA))/5)
    
    lay2dis = range(0,len(DATA)+1,int(step_plot)) # defines which layer to display given that we want to display only 5 and the nets have diferent nb of layers.
    lay2dis[-1] = len(DATA)-1
    
    # definitions for the axes
    left, width = 0.12, 0.85
    bottom, height = 0.12, 0.15
    #bottom_h = left_h = left + width + 0.01
    
    rect_histy1 = [left, bottom, width, height]
    
    #rect_histx = [left, bottom_h, width, 0.2]
    rect_histy2 = [left, bottom + height + 0.03, width, height]
    rect_histy3 = [left, bottom + height + 0.03 + height + 0.03, width, height]
    rect_histy4 = [left, bottom + height + 0.03 + height + 0.03 + height + 0.03, width, height]
    rect_histy5 = [left, bottom + height + 0.03 + height + 0.03 + height + 0.03 + height + 0.03, width, height]
    
    fig = plt.figure(figsize = (7,9))

    width_bars = Bins[1] - Bins[0] # width used for the bars of the bar plot
    
    ax1 = plt.axes(rect_histy1)
    
    #hist_hue = np.histogram( np.concatenate((ARG_HSENS1[i][M_HSENS1[i]>t2],ARG_HSENS2[i][M_HSENS2[i]>t2])),bins = Bins )
    X = DATA[lay2dis[0]][~np.isnan(DATA[lay2dis[0]])]
    hist_hue = np.histogram( X,bins = Bins )
    Y = (hist_hue[0])/X.size
    ax1.bar(hist_hue[1][:-1] - width_bars/2,Y, width = width_bars,color = 'k')
    if mean == True:
        ax1.vlines(X.mean(),0,1, color = 'red')
    
    ax2 = plt.axes(rect_histy2)
    X = DATA[lay2dis[1]][~np.isnan(DATA[lay2dis[1]])]
    hist_hue = np.histogram( X,bins = Bins )
    Y = (hist_hue[0])/X.size
    ax2.bar(hist_hue[1][:-1] - width_bars/2, Y, width  = width_bars,color = 'k')
    if mean == True:
        ax2.vlines(X.mean(),0,1, color = 'red')

    ax3 = plt.axes(rect_histy3)
    X = DATA[lay2dis[2]][~np.isnan(DATA[lay2dis[2]])]
    hist_hue = np.histogram( X,bins = Bins )
    Y = (hist_hue[0])/X.size
    ax3.bar(hist_hue[1][:-1] - width_bars/2, Y, width = width_bars,color = 'k')
    if mean == True:
        ax3.vlines(X.mean(),0,1, color = 'red')
    
    ax4 = plt.axes(rect_histy4)
    X = DATA[lay2dis[3]][~np.isnan(DATA[lay2dis[3]])]
    hist_hue = np.histogram( X,bins = Bins )
    Y = (hist_hue[0])/X.size
    ax4.bar(hist_hue[1][:-1] - width_bars/2,Y, width = width_bars,color = 'k')
    if mean == True:
        ax4.vlines(X.mean(),0, 1,color = 'red')
    
    ax5 = plt.axes(rect_histy5)
    X = DATA[lay2dis[4]][~np.isnan(DATA[lay2dis[4]])]
    hist_hue = np.histogram( X,bins = Bins )
    Y = (hist_hue[0])/X.size
    ax5.bar(hist_hue[1][:-1] - width_bars/2, Y, width = width_bars,color = 'k')
    if mean == True:
        ax5.vlines(np.mean(X),0,1, color = 'red')
    #import pdb; pdb.set_trace()
    
    #plt.setp(ax1.get_xticklabels(),fontsize=15)
    #plt.setp(ax1.get_yticklabels(),fontsize=15)
    
    #ax1.set_yticks((Bins))
    ax1.set_yticks([])
    ax1.set_xlabel(y_label)
    ax1.set_ylabel('%s' %str(lay2dis[0]))
    ax1.set_xlim([0 - width_bars/2,Bins[-1]- width_bars/2])
    
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_ylabel('%s' %str(lay2dis[1]))
    ax2.set_xlim([0 - width_bars/2,Bins[-1]- width_bars/2])
    
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_ylabel('%s' %str(lay2dis[2]))
    ax3.set_xlim([ 0 - width_bars/2,Bins[-1]- width_bars/2])
    
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_ylabel('%s' %str(lay2dis[3]))
    ax4.set_xlim([0- width_bars/2,Bins[-1]- width_bars/2])
    
    ax5.set_yticks([])
    ax5.set_xticks([])
    ax5.set_ylabel('%s' %str(lay2dis[4]))
    ax5.set_xlim([0- width_bars/2,Bins[-1]- width_bars/2])
    
    fig.text(0.04, 0.52, 'Layer', ha='center',rotation = 90)
    
    #fig.tight_layout()
    plt.show()
    fig.savefig('../figures/' + title_fig + '_' + name_net,dpi = 300)
    plt.close() 
