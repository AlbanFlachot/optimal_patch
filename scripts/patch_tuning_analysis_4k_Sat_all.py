from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import pickle


import sys
sys.path.append('../')

from utils import FUNCTIONS as F # script with a bunch of functions
from utils import DISPLAYS as D # script with functions to display


	## Fit functions
	#____________________________________________________________________________________________________________________

    
# In[9]:
###__________________________________________________________________________________________________________________________________


#with open(caffe_root +'my_session/LIBRARY/MAX_Outputs/VGG-19patch_val.pickle') as r:  # Python 3: open(..., 'wb')
#	PATCH = pickle.load(r)[2]

name_net = 'VGG-19'

print('We start analysis for net: ' + name_net)

if name_net == 'VGG-19':
    with open('../pickles/VGG-19_patches_4k_Sat.pickle') as r:  # Python 3: open(..., 'wb')
        MAX, GREY,MM1,MM2,MM3,MM4,ROT,CLASS,Good_Class= pickle.load(r)
    with open('../pickles/VGG-19_patches_GREY_ROT.pickle') as r:  # Python 3: open(..., 'wb')
        GREY_whole, ROT_whole, CLASS_whole, CLASS_GREY, Good_Class_whole = pickle.load(r)

elif name_net == 'VGG-16':
    with open('../pickles/VGG-16_patches_4k_Sat.pickle') as r:  # Python 3: open(..., 'wb')
        MAX, GREY,MM1,MM2,MM3,MM4,ROT,CLASS,Good_Class= pickle.load(r)
    with open('../pickles/VGG-16_patches_GREY_ROT.pickle') as r:  # Python 3: open(..., 'wb')
        GREY_whole, ROT_whole, CLASS_whole, CLASS_GREY, Good_Class_whole = pickle.load(r)

elif name_net == 'AlexNet':
    with open('../pickles/AlexNet_patches_4k_Sat.pickle') as r:  # Python 3: open(..., 'wb')
        MAX, GREY,MM1,MM2,MM3,MM4,ROT,CLASS,Good_Class = pickle.load(r)
    with open('../pickles/AlexNet_patches_GREY_ROT.pickle') as r:  # Python 3: open(..., 'wb')
        GREY_whole, ROT_whole, CLASS_whole, CLASS_GREY, Good_Class_whole = pickle.load(r)


RESP = list()
C2A = list()
M_HSENS1 = list() # sensitivity in seg 1
M_HSENS2 = list() # sensitivity in seg 2
M_HSENS3 = list() # sensitivity in seg 3
M_HSENS4 = list() # sensitivity in seg 4
M_ROT = list() # sensitivity to hue rotations
SENS_SAT = list() # saturation sensitivity
HUE_SAT = list() # hue change with saturation
M_HSENSALL = list() # hue sensitivity all seg
SENSITIVITY = list() # General color sensitivity

ARG_HSENS1 = list()
ARG_HSENS2 = list()
ARG_HSENS3 = list()
ARG_HSENS4 = list()

CLASS2 = list(CLASS)
CATEGORY = list(CLASS)
for l in range(0,len(CLASS)):
	CLASS2[l] = CLASS[l].copy()
	for k in range(0,len(CLASS2[l])):
		CLASS2[l][k] = (CLASS2[l][k] == CLASS2[l][k,0,2])*1
		CLASS[l][k] = (CLASS[l][k] == Good_Class[l][k])*1

  
CLASS2_whole = list(CLASS_whole)
CATEGORY_whole = list(CLASS_whole)
for l in range(0,len(CLASS_whole)):
	CLASS2_whole[l] = CLASS_whole[l].copy()
	for k in range(0,len(CLASS2_whole[l])):
		CLASS2_whole[l][k] = (CLASS2_whole[l][k] == CLASS2_whole[l][k,0])*1
		CLASS_whole[l][k] = (CLASS_whole[l][k] == Good_Class[l][k])*1
		CLASS_GREY[l][k] = (CLASS_GREY[l][k] == Good_Class[l][k])*1

for l in range(0,len(MM1)):
	MM1[l][np.isinf(MM1[l])] = 1
	MM2[l][np.isinf(MM2[l])] = 1
	MM3[l][np.isinf(MM1[l])] = 1
	MM4[l][np.isinf(MM2[l])] = 1
	ROT[l][np.isinf(ROT[l])] = 1
	MM1[l][np.isnan(MM1[l])] = 1
	MM3[l][np.isnan(MM3[l])] = 1
	MM2[l][np.isnan(MM2[l])] = 1
	MM4[l][np.isnan(MM4[l])] = 1
	ROT[l][np.isnan(ROT[l])] = 1
	MM1[l][MM1[l] == 0] = 1
	MM2[l][MM2[l] == 0] = 1
	ROT[l][ROT[l] == 0] = 1
	T = np.nanmax((MM1[l],MM2[l],MM3[l],MM4[l]),axis = (0,-1)) #max across partial kernels and hue
	Tpk = np.nanmax((MM1[l],MM2[l],MM3[l],MM4[l]),axis = -1) # max across hue
	P = np.nanmin((MM1[l],MM2[l],MM3[l],MM4[l]),axis = (0,-1)) # min across kernels and hue
	BIG_MAX = np.amax((np.amax(T,axis = 1),np.amax(ROT_whole[l],axis = 1),GREY[l] ),axis =0)
	#SENSITIVITY.append(np.amax((np.amax(T,axis = 1),np.amax(ROT_whole[l],axis = 1),GREY[l] ),axis =0)- np.amin((np.amin(T,axis = 1),np.amin(ROT_whole[l],axis = 1),GREY[l] ),axis =0))/T)
	SENSITIVITY.append((T-P)/T)
	RESP.append(((T.T-GREY[l]).T)/T)
	C2A.append((T.T-GREY[l]).T/(T.T+GREY[l]).T)
	#RESP.append((MAX[l] - GREY[l])/MAX[l])
	M_HSENS1.append((np.nanmax(MM1[l],axis = -1)-np.nanmin(MM1[l],axis = -1))/np.nanmax(MM1[l],axis = -1))
	M_HSENS1[l][np.isnan(M_HSENS1[l])] = 0
	M_HSENS2.append((np.nanmax(MM2[l],axis = -1)-np.nanmin(MM2[l],axis = -1))/np.nanmax(MM2[l],axis = -1))
	M_HSENS2[l][np.isnan(M_HSENS2[l])] = 0
	M_HSENS3.append((np.nanmax(MM3[l],axis = -1)-np.nanmin(MM3[l],axis = -1))/np.nanmax(MM3[l],axis = -1))
	M_HSENS3[l][np.isnan(M_HSENS3[l])] = 0
	M_HSENS4.append((np.nanmax(MM4[l],axis = -1)-np.nanmin(MM4[l],axis = -1))/np.nanmax(MM4[l],axis = -1))
	M_HSENS4[l][np.isnan(M_HSENS4[l])] = 0
	M_ROT.append((np.nanmax(ROT[l],axis = -1)-np.nanmin(ROT[l],axis = -1))/np.nanmax(ROT[l],axis = -1))
	hsensall = np.array([M_HSENS1[l],M_HSENS1[l],M_HSENS1[l],M_HSENS1[l]])
	hsensall = np.moveaxis(hsensall,0,-1)
	M_HSENSALL.append(hsensall)
	#SensSAT = 1 - np.nanmin(hsensall[:,1:],axis = 1)/np.nanmax(hsensall[:,1:],axis = 1) # change in sensitivity with saturation
	SensSAT = 1 - np.nanmin(Tpk[:,:,:],axis = -1)/np.nanmax(Tpk[:,:,:],axis = -1) # change in max response with saturation
	RespSAT = 1 - np.nanmin(T,axis = -1)/np.nanmax(T,axis = -1) # Saturation responsivity
	#Sat1 = 1 - np.amin()
	#M_SAT.append()
	ARG_HSENS1.append(np.nanargmax(MM1[l],axis = -1)*15)
	ARG_HSENS2.append(np.nanargmax(MM2[l],axis = -1)*15)
	ARG_HSENS3.append(np.nanargmax(MM3[l],axis = -1)*15)
	ARG_HSENS4.append(np.nanargmax(MM4[l],axis = -1)*15)
	argall = np.array([ARG_HSENS1[l], ARG_HSENS2[l], ARG_HSENS3[l], ARG_HSENS4[l]])
	HueSat = np.absolute((((argall[:,:,-1] - argall[:,:,1]) + 180)%360 - 180))
	SENS_SAT.append(SensSAT)
	HUE_SAT.append(np.moveaxis(HueSat,0,-1))

t1 = 0.25
t2 = 0.5
t3 = 0.75


# In[9]:


# In[9]:

#-------------------------------------------------------------------------------------------------------------
#### RESPONSIVITY

respt1,respt2,respt3,Mean_resp,Std_resp = F.RESPO(RESP,t1,t2,t3)

D.PLOT_FIGURE('CR',respt1,respt2,respt3,Mean_resp)


D.DEFINE_PLT_RC(type = 0.33)
	# Plot distribution Responsivity
Treshs = np.array([0,1/8,2/8,3/8,4/8,5/8,6/8,7/8])
#D.PLOT_FIGURE_GRADUATE_DISTRIB(RESP,Treshs,'CR')
DIS_CR = F.DISTRIB_resp(RESP,Treshs)
D.PLOT_FIGURE_GRADUATE_DISTRIB(DIS_CR,'CR')

D.plot_fig_summary(respt2,Mean_resp)

# In[9]:


# plot global sensitivity proportions
DIS_OCS = F.DISTRIB_resp(SENSITIVITY,Treshs, resp_type = 1)
D.PLOT_FIGURE_GRADUATE_DISTRIB(DIS_OCS,'OCS')

print('Proportion of weakly color sensitive kernels in first layer is %f' %(DIS_OCS[0,0]-DIS_OCS[0,1]))
print('Proportion of highly color sensitive kernels in first layer is %f' %(DIS_OCS[0,-1]))

#D.PLOT_FIGURE_GRADUATE_DISTRIB(SENSITIVITY,Treshs,' OCS', resp_type = 1)

SENSt1,SENSt2,SENSt3,Mean_SENS,Std_SENS = F.RESPO(SENSITIVITY,t1,t2,t3)
D.plot_fig_summary(SENSt2,Mean_SENS)


#-------------------------------------------------------------------------------------------------------------
#### CS_HSENS1

selecMM1t1,selecMM1t2,selecMM1t3,Mean_selec_HSENS1,Std_selec_HSENS1 = F.RESPO(M_HSENS1,t1,t2,t3)
#PLOT_FIGURE('CS1',selecMM1t1,selecMM1t2,selecMM1t3,Mean_selec_HSENS1)


#-------------------------------------------------------------------------------------------------------------
#### CS_HSENS2
selecMM2t1,selecMM2t2,selecMM2t3,Mean_selec_HSENS2,Std_selec_HSENS2 = F.RESPO(M_HSENS2,t1,t2,t3)
#PLOT_FIGURE('CS2',selecMM2t1,selecMM2t2,selecMM2t3,Mean_selec_HSENS2)

#-------------------------------------------------------------------------------------------------------------
#### CS_HSENS3

selecMM3t1,selecMM3t2,selecMM3t3,Mean_selec_HSENS3,Std_selec_HSENS3 = F.RESPO(M_HSENS3,t1,t2,t3)
#PLOT_FIGURE('CS3',selecMM3t1,selecMM3t2,selecMM3t3,Mean_selec_HSENS3)


#-------------------------------------------------------------------------------------------------------------
#### CS_HSENS4
selecMM4t1,selecMM4t2,selecMM4t3,Mean_selec_HSENS4,Std_selec_HSENS4 = F.RESPO(M_HSENS4,t1,t2,t3)
#PLOT_FIGURE('CS4',selecMM4t1,selecMM4t2,selecMM4t3,Mean_selec_HSENS4)


#-------------------------------------------------------------------------------------------------------------
#### CS_rot


selecrott1,selecrott2,selecrott3,Mean_selec_rot,Std_selec_rot = F.RESPO(M_ROT,t1,t2,t3)

#PLOT_FIGURE('CSROT',selecrott1,selecrott2,selecrott3,Mean_selec_rot)


	# Plot distribution ROT
Treshs = np.array([0,1/8,2/8,3/8,4/8,5/8,6/8,7/8])
DIS_M = np.zeros((len(CLASS),len(Treshs)))

MEAN_M = np.zeros((len(CLASS)))

count = 0
for l in range(0,len(M_ROT)):
	M = np.amax(np.stack((M_HSENS1[l],M_HSENS2[l],M_HSENS3[l],M_HSENS4[l]),axis = 1),axis = -1)
	MEAN_M[l] = np.mean(M)
	DIS_M[l,0] = F.respo(M,Treshs[0])*100
	DIS_M[l,1] = F.respo(M,Treshs[1])*100
	DIS_M[l,2] = F.respo(M,Treshs[2])*100
	DIS_M[l,3] = F.respo(M,Treshs[3])*100
	DIS_M[l,4] = F.respo(M,Treshs[4])*100
	DIS_M[l,5] = F.respo(M,Treshs[5])*100
	DIS_M[l,6] = F.respo(M,Treshs[6])*100
	DIS_M[l,7] = F.respo(M,Treshs[7])*100
	count +=1

D.PLOT_FIGURE_GRADUATE_DISTRIB(DIS_M,'HS') 
 


'''
SUM_sens = np.zeros(len(ROT))
for i in range(len(ROT)):
	SUM_sens[i] = np.sum(  (M_HSENS1[i] >t2) | (M_HSENS2[i] >t2) | (M_HSENS3[i] >t2) | (M_HSENS4[i] >t2) )/M_HSENS4[i].size
	#np.sum(M_ROT[i])/M_HSENS4[i].size

#plot_fig_summary(SUM_sens,MEAN_M)'''


# In[9]: Proportion of saturation sensitive kernels

DIS_CCR = F.DISTRIB_resp(SENS_SAT,Treshs, resp_type = 0)
D.PLOT_FIGURE_GRADUATE_DISTRIB(DIS_CCR,'CCR')



senssatt1,senssatt2,senssatt3,Mean_senssat,Std_senssat = F.RESPO(SENS_SAT,t1,t2,t3, resp_type = 0)
D.plot_fig_summary(senssatt2,Mean_senssat)


# compute correlatiosn between color sensitivity and hue selectivity of kernels
corr_OCS_HS = np.zeros(len(CLASS))
corr_OCS_CCR = np.zeros(len(CLASS))
corr_HS_CCR = np.zeros(len(CLASS))
for l in range(len(CLASS)):
	M = np.amax(np.stack((M_HSENS1[l],M_HSENS2[l],M_HSENS3[l],M_HSENS4[l]),axis = 1),axis = -1)
	corr_OCS_HS[l] = np.corrcoef(np.amax(SENSITIVITY[l],axis = -1),np.amax(M,axis = -1))[0,1]
	corr_OCS_CCR[l] = np.corrcoef(np.amax(SENSITIVITY[l],axis = -1),np.amax(SENS_SAT[l],axis = 0))[0,1]
	corr_HS_CCR[l] = np.corrcoef(np.amax(M,axis = -1),np.amax(SENS_SAT[l],axis = 0))[0,1]

# In[9]:
#-------------------------------------------------------------------------------------------------------------
#### Proportion of color selective kernels

ARG_SEL = list()
Nb_col_select = list()
for l in range(len(M_HSENS1)):
	nb_kernels = len(M_HSENS1[l])
	P_sel = np.array([np.amax(M_HSENS1[l], axis = -1) > t2, np.amax(M_HSENS2[l], axis = -1) > t2, np.amax(M_HSENS3[l], axis = -1) > t2, np.amax(M_HSENS4[l], axis = -1) > t2])
	Nb_col_select.append( np.sum(P_sel,axis = 0))
	Arg_sel = np.zeros(P_sel.shape)
	Arg_sel[:] = np.nan
	Arg_sel[P_sel] = np.array([ARG_HSENS1[l][range(len(ARG_HSENS1[l])),np.argmax(M_HSENS1[l], axis = -1)], ARG_HSENS2[l][range(len(ARG_HSENS2[l])),np.argmax(M_HSENS2[l], axis = -1)], ARG_HSENS3[l][range(len(ARG_HSENS3[l])),np.argmax(M_HSENS3[l], axis = -1)], ARG_HSENS4[l][range(len(ARG_HSENS4[l])),np.argmax(M_HSENS4[l], axis = -1)]])[P_sel]
	#ARG_SEL.append(Arg_sel)
	for i in range(nb_kernels):
		for j in range(len(Arg_sel[:,i])-1):
			start = Arg_sel[j,i]
			if np.isnan(Arg_sel[j,i].any()):
				continue
			else:
				for k in range(j+1,len(Arg_sel[:,i])):
					if np.isnan(Arg_sel[k,i].any()):
						continue
					else:
						if np.absolute(start - Arg_sel[k,i]) <= 30:
							Arg_sel[j,i] = np.nanmean((start,Arg_sel[k,i]))
							Arg_sel[k,i] = np.nan
							Nb_col_select[l][i] -= 1
	#Nb_col_select.append( np.sum(P_sel,axis = 0))
	ARG_SEL.append(Arg_sel)

print np.sum(Nb_col_select[-1]==4)

nb_lay = len(M_HSENS1)
Theta = np.arange(0,2*np.pi,2*np.pi/nb_lay)
Sat = 0.43
Lum = 0.2

(x,y) = F.pol2cart(Sat, Theta)

color_id = F.PCA2RGB(np.array([[np.zeros(nb_lay)+Lum],[x],[y]]).T)+0.5
#color_id = PCA2RGB(np.array([[Lum],[np.zeros(nb_lay)],[np.zeros(nb_lay)]]).T)+0.55
color_id = color_id.reshape((len(color_id),3))

D.DEFINE_PLT_RC(type = 0.5)

fig = plt.figure(1, figsize=(7, 5))

rect_ax1 = [0.1, 0.11, 0.69, 0.85]
#rect_ax2 = [LEFT+WIDTH+2*LEFT, bottom + height+0.1, WIDTH, HEIGHT]

Bins = np.arange(0,6,1)

ax1 = plt.axes(rect_ax1)
count =1
for i in range(0,nb_lay,2):
	#hist_hue = np.histogram( np.concatenate((ARG_HSENS1[i][M_HSENS1[i]>t2],ARG_HSENS2[i][M_HSENS2[i]>t2])),bins = Bins )
	h = np.histogram( Nb_col_select[i],bins = Bins )
	ax1.plot( h[1][:-1], (h[0])/len(Nb_col_select[i]),linestyle = '-',color = color_id[i],label = 'Layer %s' %str(i+1),linewidth = 2)
	count +=1


ax1.set_xticks(Bins)
ax1.set_xlim([0,4])

fig.text(0.45, 0.02, 'Nb colors', ha='center',fontsize = 15)
ax1.legend(bbox_to_anchor=(1.01, 0, 0.275, 0.95), loc=1,
           ncol=1, mode="expand", borderaxespad=0.,fontsize=12)

#fig.tight_layout()





# In[9]:

#### Horizontal histograms of hue tuning ------------------------------------------------------------------------------

D.DEFINE_PLT_RC(type = 0.5)



D.plot_horizontal_histo(Nb_col_select, np.arange(0,6,1), 'Number of hues', 'hue_count_t2', name_net)
D.plot_vertical_histo(Nb_col_select, np.arange(0,6,1), 'Number of hues', 'hue_count_t2', name_net, mean = True)


# In[9]:

#### Horizontal histograms of hue tuning ------------------------------------------------------------------------------

D.DEFINE_PLT_RC()

D.plot_horizontal_histo(ARG_SEL, np.arange(0,365,15), 'Azimuth (degrees)', 'histo_preferred_hues',name_net)
D.plot_vertical_histo(ARG_SEL, np.arange(0,365,15), 'Azimuth (degrees)', 'histo_preferred_hues',name_net)


# In[9]:

#-------------------------------------------------------------------------------------------------------------
#### Correlation and proportions CS_HSENS1, CS_HSENS2

Corrt1 = np.zeros(len(CLASS))
Corrt2 = np.zeros(len(CLASS))
Corrt3 = np.zeros(len(CLASS))
Corrt4 = np.zeros(len(CLASS))
Corrt5 = np.zeros(len(CLASS))
for l in range(0,len(RESP)):
	Corrt1[l] = np.corrcoef(M_HSENS1[l],M_HSENS2[l])[0,1]
	Corrt2[l] = np.sum((M_HSENS1[l] > t2)*1)/len(M_HSENS1[l])
	Corrt5[l] = np.sum((M_HSENS2[l] > t2)*1)/len(M_HSENS1[l])
	Corrt3[l] = np.sum(((M_HSENS1[l] > t2)& (M_HSENS2[l] > t2)) *1)/len(M_HSENS1[l])
	Corrt4[l] = np.sum(((M_HSENS1[l] > t2)| (M_HSENS2[l] > t2)) *1)/len(M_HSENS1[l])

np.sum((M_HSENS1[l] > t2)*1)/len(M_HSENS1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(1,len(CLASS)+1),Corrt1,'k-',linewidth = 4,label = 'Tresh = 0.25')
#ax.plot(np.arange(1,6),Corrt2,'k--',linewidth = 4,label = 'Tresh = 0.50')
#ax.plot(np.arange(1,6),Corrt3,'k-',linewidth = 4,label = 'Tresh = 0.75')
#ax.plot(np.arange(1,8),Comp,'o',ms = 12,label = '1-<r$^2$>$^2$')
plt.setp(ax.get_xticklabels(), fontsize=20)
plt.setp(ax.get_yticklabels(), fontsize=20)
plt.xlabel('Layer',fontsize = 25)
plt.ylabel('Correlation (CS$_{MM1}$,CS$_{MM2}$)',fontsize = 25)
plt.ylim(0,1)
plt.xlim(0.5,len(CLASS)+0.5)
#plt.legend(loc='upper center', fontsize = 15)
#fig.tight_layout()
plt.show()
#fig.savefig('tuning_curves/BVLC: Responsivity (normalized)')

plt.close(fig)



# In[9]:

#-______________________________________________________________________________________________________-

#### MISCLASSIFICATION

	#-----------------------------------------------------------------------------------------------
	## Some proportions



Class_rot2 = list()


C_HSENS1sur = np.zeros(len(CLASS))

for l in range(0,len(RESP)):
	C2 = np.zeros(CLASS[l].shape[0])
	for k in range(0,len(CLASS[l])):
		C2[k] = CLASS[l][k,-1,0,4]

	Class_rot2.append(C2)

Class_HSENS1, C_HSENS1, C_HSENS1_tot = F.misclass(CLASS, M_HSENS1,t2,0)
Class_HSENS2, C_HSENS2, C_HSENS2_tot = F.misclass(CLASS, M_HSENS2,t2,1)
Class_HSENS3, C_HSENS3, C_HSENS3_tot = F.misclass(CLASS, M_HSENS3,t2,2)
Class_HSENS4, C_HSENS4, C_HSENS4_tot = F.misclass(CLASS, M_HSENS4,t2,3)
Class_rot, C_rot, C_rot_tot = F.misclass(CLASS, M_ROT,t2,4)


# In[9]:

HUE_as_axis = np.arange(0,360,15)
    
#-----------------------------------------------------------------------------------------------
	## Proportions of classes correctly classified originally wrongly classified after color change

 
prop_HSENS1, prop_non_HSENS1 = F.prop_misclass(CLASS, M_HSENS1, t2,0)
prop_HSENS2, prop_non_HSENS2 = F.prop_misclass(CLASS, M_HSENS2, t2,1)
prop_HSENS3, prop_non_HSENS3 = F.prop_misclass(CLASS, M_HSENS3, t2,2)
prop_HSENS4, prop_non_HSENS4 = F.prop_misclass(CLASS, M_HSENS4, t2,3)
prop_ROT, prop_non_ROT = F.prop_misclass(CLASS, M_ROT, t2,4)

prop_miss_SENS, prop_miss_non_SENS, prop_miss_all =  F.prop_misclass_all(CLASS, SENSITIVITY, t2)

'''DEFINE_PLT_RC()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(1,len(CLASS)+1),prop_HSENS1*100,'r',label = 'Color selective,\nsegment 1')
ax.plot(np.arange(1,len(CLASS)+1),prop_HSENS2*100,'g',label = 'Color selective,\nsegment 2')
ax.plot(np.arange(1,len(CLASS)+1),prop_HSENS3*100,'orange',label = 'Color selective,\nsegment 3')
ax.plot(np.arange(1,len(CLASS)+1),prop_HSENS4*100,'blue',label = 'Color selective,\nsegment 4')
#ax.plot(np.arange(1,len(CLASS)+1),prop_ROT*100,'k',linewidth = 4,label = 'Color selective,\nmodif C')
#ax[i-3].plot(HUE_as_axis,Classification2_t2[i,:,2]*100,'k',linewidth = 4,label = 'Color selective,\nmodif C')
#ax[np.unravel_index(i,(2,3))].plot(HUE_as_axis,Classification2[i,:,2]*100,'grey',linewidth = 4,label = 'All units')
ax.plot(np.arange(1,len(CLASS)+1),prop_non_HSENS1*100,'r',linestyle = '--',label = 'Non color selective,segment 1')
ax.plot(np.arange(1,len(CLASS)+1),prop_non_HSENS2*100,'g',linestyle = '--',label = 'Non color selective, segment 2')
ax.plot(np.arange(1,len(CLASS)+1),prop_non_HSENS3*100,'orange',linestyle = '--',label = 'Non color selective, segment 3')
ax.plot(np.arange(1,len(CLASS)+1),prop_non_HSENS4*100,'b',linestyle = '--',label = 'Non color selective, segment 4')
#ax.plot(np.arange(1,len(CLASS)+1),prop_non_C*100,'k',linestyle = '--',linewidth = 4,label = 'Non color selective, modif C')
#ax[i-3].plot(HUE_as_axis,Classification2[i,:,2]*100,'k--',linewidth = 4,label = 'Non color selective, modif C')
#ax.plot(np.arange(1,8),Comp,'o',ms = 12,label = '1-<r$^2$>$^2$')
plt.setp(ax.get_xticklabels())
plt.setp(ax.get_yticklabels())
plt.xlabel('Layer')
plt.ylabel('Proportion of kernels')
plt.ylim(0,100)
plt.xlim(0.5,len(CLASS)+0.5)
#plt.legend(loc='upper center', fontsize = 20)
fig.tight_layout()
plt.show()
#fig.savefig('tuning_curves/BVLC: Responsivity (normalized)')
plt.close(fig)'''

D.DEFINE_PLT_RC()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(1,len(CLASS)+1),prop_miss_non_SENS*100,color = [0.5, 0.5, 0.5],linestyle = '-',label = 'Non color selective,segment 1')
ax.plot(np.arange(1,len(CLASS)+1),prop_miss_SENS*100,'k',label = 'Color selective,\nsegment 1')
ax.plot(np.arange(1,len(CLASS)+1),prop_miss_all*100,'r',label = 'Color selective,\nsegment 1')
plt.xlabel('Layer')
plt.ylabel('Proportion of images (%)')
plt.ylim(0,100)
plt.xlim(0.5,len(CLASS)+0.5)
#plt.legend(loc='upper center', fontsize = 20)
fig.tight_layout()
plt.show()
fig.savefig('../figures/from_correct_to_incorrect_' + name_net, dpi = 300)
plt.close(fig)


print('Proportion of non color senstitive kernels that go from correctly to incorrectly classified is %f ' %prop_miss_non_SENS[-1])
print('Proportion of color senstitive kernels that go from correctly to incorrectly classified is %f ' %prop_miss_SENS[-1])
print('Proportion of all kernels that go from correctly to incorrectly classified is %f ' %prop_miss_all[-1])

# In[9]:
	#-----------------------------------------------------------------------------------------------
		## Proportions of correct classifications as function of modification angle




### Most and least misclassified classes originally correctly classified
MISCLASS_HSENS1 = list()
MISCLASS_HSENS2 = list()
MISCLASS_ROT = list()
MISCLASS = list()
for l in range(0,len(CLASS)):
	MISCLASS_HSENS1.append(np.sum(CLASS[l][:,:,0]==0,axis = (1)))
	MISCLASS_HSENS2.append(np.sum(CLASS[l][:,:,1]==0,axis = (1)))
	MISCLASS_ROT.append(np.sum(CLASS[l][:,:,4]==0,axis = (1)))
	MISCLASS_HSENS1[l][CLASS[l][:,0,4] == 0] = 12
	MISCLASS_HSENS2[l][CLASS[l][:,0,4] == 0] = 12
	MISCLASS_ROT[l][CLASS[l][:,0,4] == 0] = 12
	MISCLASS.append(np.sum(CLASS[l]==0,axis = (1,2)))
	MISCLASS[l][CLASS[l][:,0,4] == 0] = 36

MISCLASS[l].argsort()[-5:][::-1]
#print(labels[(Good_Class[4][MISCLASS_HSENS1[4].argsort()[-5:][::-1]]).astype(int)])
#print(labels[(Good_Class[4][MISCLASS_HSENS1[4].argsort()[:5][::-1]]).astype(int)])
#print(labels[(Good_Class[4][MISCLASS_HSENS2[4].argsort()[-5:][::-1]]).astype(int)])
#print(labels[(Good_Class[4][MISCLASS_HSENS2[4].argsort()[:5][::-1]]).astype(int)])
#print(labels[(Good_Class[4][MISCLASS_ROT[4].argsort()[-5:][::-1]]).astype(int)])
#print(labels[(Good_Class[4][MISCLASS_ROT[4].argsort()[:5][::-1]]).astype(int)])
#print(labels[(Good_Class[4][MISCLASS[4].argsort()[-15:][::-1]]).astype(int)])
#print(labels[(Good_Class[4][MISCLASS[4].argsort()[:15][::-1]]).astype(int)])



# In[9]:
#-----------------------------------------------------------------------------------------------
### Classification as a function of hue


# loop to put the starting hue as the argmax hue

CLASS3 = list(CLASS)
for l in range(0,len(CLASS)):
	CLASS3[l] = CLASS[l].copy()
	for k in range(len(CLASS[l])):
		for h in range(CLASS[l].shape[-2]):
			argmm1 = ((ARG_HSENS1[l][k,-1]/15+h)%len(CLASS3[l][k,-1,:,0])).astype(int)
			argmm2 = ((ARG_HSENS2[l][k,-1]/15+h)%len(CLASS3[l][k,-1,:,0])).astype(int)
			argmm3 = ((ARG_HSENS3[l][k,-1]/15+h)%len(CLASS3[l][k,-1,:,0])).astype(int)
			argmm4 = ((ARG_HSENS4[l][k,-1]/15+h)%len(CLASS3[l][k,-1,:,0])).astype(int)
			CLASS3[l][k,:,h,0] = CLASS[l][k,:,argmm1,0]
			CLASS3[l][k,:,h,1] = CLASS[l][k,:,argmm2,1]
			CLASS3[l][k,:,h,2] = CLASS[l][k,:,argmm3,2]
			CLASS3[l][k,:,h,3] = CLASS[l][k,:,argmm4,3]

SENSlast = np.stack((M_HSENS1[-1],M_HSENS2[-1],M_HSENS3[-1],M_HSENS4[-1]))
arg_max_sens = np.argmax(SENSlast,axis = 0)



Classification = np.zeros((len(CLASS), CLASS[0].shape[1], CLASS[0].shape[2], 5)) # all kernels
Classification2 = np.zeros((len(CLASS), CLASS[0].shape[1], CLASS[0].shape[2], 5)) # non color selective
Classification_mean = np.zeros((len(CLASS), CLASS[0].shape[1], CLASS[0].shape[2])) # mean non color selective
Classification_meant2 = np.zeros((len(CLASS), CLASS[0].shape[1], CLASS[0].shape[2])) # mean color selective
Classification2_t2 = np.zeros((len(CLASS), CLASS[0].shape[1], CLASS[0].shape[2], 5)) # color selective

CLASSIFICATION_ROT = np.zeros((len(CLASS_whole), CLASS_whole[0].shape[1])) # all kernels, whole image rotation

for l in range(0,len(RESP)):
	for hue in range(0,CLASS[0].shape[-2]):
		CLASSIFICATION_ROT[l,hue] = np.sum(CLASS_whole[l][:,hue])/CLASS_whole[l][:,hue].size
		for sat in range(CLASS[0].shape[-3]):
			C_R = CLASS[l][:,0,4]
			Classification2[l,sat,hue,0] = np.sum(CLASS3[l][(M_HSENS1[l]<t2)[:,sat], sat,hue,0])/len(CLASS3[l][ (M_HSENS1[l]<t2)[:,sat]])
			Classification2[l,sat,hue,1] = np.sum(CLASS3[l][ (M_HSENS2[l]<t2)[:,sat], sat,hue,1])/len(CLASS3[l][ (M_HSENS2[l]<t2)[:,sat]])
			Classification2[l,sat,hue,2] = np.sum(CLASS3[l][ (M_HSENS3[l]<t2)[:,sat], sat,hue,2])/len(CLASS3[l][ (M_HSENS3[l]<t2)[:,sat]])
			Classification2[l,sat,hue,3] = np.sum(CLASS3[l][ (M_HSENS4[l]<t2)[:,sat], sat,hue,3])/len(CLASS3[l][ (M_HSENS4[l]<t2)[:,sat]])
			Classification2[l,sat,hue,-1] = np.sum(CLASS3[l][ (M_ROT[l]<t2)[:,sat], sat,hue,-1])/len(CLASS3[l][ (M_ROT[l]<t2)[:,sat]])

			Classification[l,sat,hue,0] = np.sum(CLASS3[l][ :,sat,hue,0])/len(CLASS3[l])
			Classification[l,sat,hue,1] = np.sum(CLASS3[l][ :,sat,hue,1])/len(CLASS3[l])
			Classification[l,sat,hue,2] = np.sum(CLASS3[l][ :,sat,hue,2])/len(CLASS3[l])
			Classification[l,sat,hue,3] = np.sum(CLASS3[l][:,sat,hue,3])/len(CLASS3[l])
			Classification[l,sat,hue,-1] = np.sum(CLASS3[l][ :,sat,hue,4])/len(CLASS3[l])


			Classification2_t2[l,sat,hue,0] = np.sum(CLASS3[l][(M_HSENS1[l]>t2)[:,sat], sat,hue,0])/len(CLASS3[l][ (M_HSENS1[l]>t2)[:,sat]])
			Classification2_t2[l,sat,hue,1] = np.sum(CLASS3[l][ (M_HSENS2[l]>t2)[:,sat], sat,hue,1])/len(CLASS3[l][ (M_HSENS2[l]>t2)[:,sat]])
			Classification2_t2[l,sat,hue,2] = np.sum(CLASS3[l][ (M_HSENS3[l]>t2)[:,sat], sat,hue,2])/len(CLASS3[l][ (M_HSENS3[l]>t2)[:,sat]])
			Classification2_t2[l,sat,hue,3] = np.sum(CLASS3[l][ (M_HSENS4[l]>t2)[:,sat], sat,hue,3])/len(CLASS3[l][ (M_HSENS4[l]>t2)[:,sat]])
			Classification2_t2[l,sat,hue,-1] = np.sum(CLASS3[l][ (M_ROT[l]>t2)[:,sat], sat,hue,-1])/len(CLASS3[l][ (M_ROT[l]>t2)[:,sat]])
       
	w = [1-selecMM1t2[l], 1-selecMM2t2[l], 1-selecMM3t2[l], 1-selecMM4t2[l]]
	Classification_mean[l] = np.average(Classification2[l,:,:,:-1], weights = w,axis = -1) # mean non color selective
	w = [selecMM1t2[l], selecMM2t2[l], selecMM3t2[l], selecMM4t2[l]]
	Classification_meant2[l] = np.average(Classification2_t2[l,:,:,:-1], weights = w, axis = -1) # mean color selective




# In[9]:
	### Figure poster relevance of color for object recognition

 

D.DEFINE_PLT_RC(type = 1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(HUE_as_axis,Classification2_t2[-1,:,:,0].T*100,'r',label = 'Color selective,\nsegment 1')
ax.plot(HUE_as_axis,Classification2_t2[-1,:,:,1].T*100,'g',label = 'Color selective,\nsegment 2')
ax.plot(HUE_as_axis,Classification2_t2[-1,:,:,2].T*100,'orange',label = 'Color selective,\nsegment 3')
ax.plot(HUE_as_axis,Classification2_t2[-1,:,:,3].T*100,'b',label = 'Color selective,\nsegment 4')
#ax[np.unravel_index(i,(2,3))].plot(HUE_as_axis,Classification2[i,:,2]*100,'grey',linewidth = 4,label = 'All units')
ax.plot(HUE_as_axis,Classification2[-1,:,:,0].T*100,'r',linestyle = '--',label = 'Color selective,\nsegment 1')
ax.plot(HUE_as_axis,Classification2[-1,:,:,1].T*100,'g',linestyle = '--',label = 'Color selective,\nsegment 2')
ax.plot(HUE_as_axis,Classification2[-1,:,:,2].T*100,'orange',linestyle = '--',label = 'Color selective,\nsegment 3')
ax.plot(HUE_as_axis,Classification2[-1,:,:,3].T*100,'b',linestyle = '--',label = 'Color selective,\nsegment 4')
#plt.setp(ax1.set_ylabel('Frequency',fontsize = 15))
plt.xlabel('Distance from preferred hue (degrees)')
plt.ylabel('Accuracy')
#plt.ylim(0,100)
#plt.xlim(0.5,len(CLASS)+0.5)
#plt.legend(loc='upper center', fontsize = 20)
fig.tight_layout()
plt.show()
#fig.savefig('tuning_curves/BVLC: Responsivity (normalized)')
plt.close(fig)


fig = plt.figure()
ax = fig.add_subplot(111)
for s in range(4):
	c = 1-(float(s)/4+0.25)
	ax.plot(HUE_as_axis,Classification_meant2[-1,1+s,:].T*100,color = [c,c,c],linewidth = 5, label = 'Color selective')
	ax.plot(HUE_as_axis,Classification_mean[-1,1+s,:].T*100,c = '%s'%str(c),linestyle = '--',linewidth = 5, label = 'Non Color selective')
#ax.plot(HUE_as_axis,np.mean(Classification[-1],axis = (0,-1)).T*100,'r',linewidth = 3)

plt.xlabel('Distance from preferred hue (degrees)')
plt.ylabel('Accuracy')
fig.tight_layout()
plt.show()
fig.savefig('../figures/classification_vs_hue_details_' + name_net, dpi=300)
plt.close(fig)

#mpl.rcParams['image.cmap'] = 'jet'
### Figure mean
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(HUE_as_axis,np.mean(Classification_mean[-1,1:], axis = 0).T*100,color = [0.5,0.5,0.5],linestyle = '-',label = 'Non color selective')
ax.plot(HUE_as_axis,np.mean(Classification_meant2[-1,1:], axis = 0).T*100,'k',linewidth = 5, label = 'Color selective')
ax.plot(HUE_as_axis,np.ones(len(HUE_as_axis))*Classification2[-1,-1,0,-1]*100,color = [0.5,0.5,0.5],linestyle = '--',label = 'Non Color selective')
ax.plot(HUE_as_axis,np.ones(len(HUE_as_axis))*Classification2_t2[-1,-1,0,-1]*100,color = [0,0,0],linestyle = '--',label = 'Color selective')
ax.plot(HUE_as_axis,np.mean(Classification[-1,1:,:,:-1],axis = (0,-1)).T*100,'r',linewidth = 4)
plt.xlabel('Distance from preferred hue (degrees)')
plt.ylabel('Accuracy')
fig.tight_layout()
plt.show()
fig.savefig('../figures/classification_vs_hue_' + name_net, dpi=300)
plt.close(fig)

print('Performance drops by %f percents after hue modification for hue selective' %((np.mean(Classification_meant2[-1,1:], axis = 0).max()-np.mean(Classification_meant2[-1,1:], axis = 0).min())/np.mean(Classification_meant2[-1,1:], axis = 0).max()))
print('Performance drops by %f percents after hue modification for non hue selective' %((np.mean(Classification_mean[-1,1:], axis = 0).max()-np.mean(Classification_mean[-1,1:], axis = 0).min())/np.mean(Classification_mean[-1,1:], axis = 0).max()))
print('Performance drops by %f percents after hue modification all kernels' %((np.mean(Classification[-1,1:,:,:-1],axis = (0,-1)).max()-np.mean(Classification[-1,1:,:,:-1],axis = (0,-1)).min())/np.mean(Classification[-1,1:,:,:-1],axis = (0,-1)).max()))




fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(HUE_as_axis,Classification2_t2[-1,:,:,-1].T*100,'k',linestyle = '-')
plt.xlabel('Distance from preferred hue (degrees)')
plt.ylabel('Accuracy')
fig.tight_layout()
plt.show()
plt.close(fig)


## Figure classification as function of hue with saturation

fig = plt.figure()
ax = fig.add_subplot(111)
for s in range(5):
	c = 1-(float(s)/6+0.3)
	ax.plot(HUE_as_axis, np.mean(Classification[-1,s,:,:-1], axis = -1)*100,color = [c,c,c],linestyle = '-')
#ax.plot(HUE_as_axis,np.mean((Classification2_t2[-1,:,0],Classification2_t2[-1,:,1],Classification2_t2[-1,:,2],Classification2_t2[-1,:,3]),0)*100,'k',linestyle = '--',linewidth = 6,label = 'mean')
ax.plot(HUE_as_axis,np.ones(len(HUE_as_axis))*Classification[-1,-1,0,-1]*100,color = [0,0,0],linestyle = '--')
plt.xlabel('Distance from preferred hue (degrees)')
plt.ylabel('Accuracy')
plt.ylim(60,95)
#plt.xlim(0.5,len(CLASS)+0.5)
#plt.legend(loc='upper center', fontsize = 20)
fig.tight_layout()
plt.show()
fig.savefig('../figures/classification_vs_hue_details_sat_' + name_net, dpi=300)       # higher res outputs)
plt.close(fig)

print('Performance drops by %f percents after hue modification 0.25 chroma' %(( np.mean(Classification[-1,1,:,:-1], axis = -1).max() - np.mean(Classification[-1,1,:,:-1], axis = -1).min() )/np.mean(Classification[-1,1,:,:-1], axis = -1).max()))
print('Performance drops by %f percents after hue modification 1 chroma' %(( np.mean(Classification[-1,-1,:,:-1], axis = -1).max() - np.mean(Classification[-1,-1,:,:-1], axis = -1).min() )/np.mean(Classification[-1,-1,:,:-1], axis = -1).max()))


# Figure rotation whole image

WEIGHTS_CLASSIFICTION = [len(CLASS[l]) for l in range(len(CLASS))]

MEAN_CLASSIFICATION_ROT = np.average(CLASSIFICATION_ROT,weights = WEIGHTS_CLASSIFICTION, axis = 0)

CLASSIFICATION_GREY = [np.sum(CLASS_GREY[l])*100/CLASS_GREY[l].size for l in range(len(CLASS))]

MEAN_CLASSIFICATION_GREY = np.average(CLASSIFICATION_GREY,weights = WEIGHTS_CLASSIFICTION)
                       
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(HUE_as_axis,CLASSIFICATION_ROT[-1]*100,'k',linestyle = '-')
ax.plot(HUE_as_axis,MEAN_CLASSIFICATION_ROT*100,'k',linestyle = '-')
#ax.axhline(y=np.sum(CLASS_GREY[-1])*100/CLASS_GREY[-1].size, color='grey', linestyle='-')
ax.axhline(y=MEAN_CLASSIFICATION_GREY,color = 'grey',linestyle = '-')
plt.xlabel('Rotation angle (degrees)')
plt.ylabel('Accuracy')
fig.tight_layout()
plt.show()
fig.savefig('../figures/classification_whole_image_' + name_net, dpi=300)       # higher res outputs)
plt.close(fig)


print('Performance drops by %f percents after conversion to B&W' %((MEAN_CLASSIFICATION_ROT[0]*100-MEAN_CLASSIFICATION_GREY)/MEAN_CLASSIFICATION_ROT[0]))
print('Performance drops by %f percents after hue modification' %((MEAN_CLASSIFICATION_ROT.max()-MEAN_CLASSIFICATION_ROT.min())/MEAN_CLASSIFICATION_ROT.max()))


# In[9]:


### ---------------------------------------------------------------------------------------------------------------------------------
# Peaks detection
#




P_HSENS1 = list()
P_HSENS2 = list()
P_HSENS3 = list()
P_HSENS4 = list()
P_ROT = list()

#prop_comp_HSENS1= np.zeros((len(CLASS),2))
#prop_comp_HSENS2= np.zeros((len(CLASS),2))
#prop_comp_rot= np.zeros((len(CLASS),2))
for l in range(0,len(CLASS)):
	Result_peak_HSENS1 = np.zeros(CLASS[l].shape[:2]+tuple([3]))
	Result_peak_HSENS2 = np.zeros(CLASS[l].shape[:2]+tuple([3]))
	Result_peak_HSENS3 = np.zeros(CLASS[l].shape[:2]+tuple([3]))
	Result_peak_HSENS4 = np.zeros(CLASS[l].shape[:2]+tuple([3]))
	Result_peak_rot = np.zeros(CLASS[l].shape[:2]+tuple([3]))
	p_rot = np.zeros((len(CLASS[l])))
	p_HSENS1 = np.zeros((len(CLASS[l])))
	p_HSENS2 = np.zeros((len(CLASS[l])))
	for K in range(len(CLASS[l])):
		for sat in range(CLASS[l].shape[1]):
			y = MM1[l][K,sat,:]
			Result_peak_HSENS1[K,sat] = F.peak_detection(y)
			y = MM2[l][K,sat,:]
			Result_peak_HSENS2[K,sat] = F.peak_detection(y)
			y = MM3[l][K,sat,:]
			Result_peak_HSENS3[K,sat] = F.peak_detection(y)
			y = MM4[l][K,sat,:]
			Result_peak_HSENS4[K,sat] = F.peak_detection(y)
			y = ROT[l][K,sat,:]
			Result_peak_rot[K,sat] = F.peak_detection(y)
		#plt.plot(Hue*180/np.pi,y,linewidth = 1,color = 'blue')
		#plt.plot(Hue*180/np.pi,yhat, color='red')
		#plt.show()
	#prop_comp_HSENS1[l] = simple_comp(Result_fit_HSENS1)
	#prop_comp_HSENS2[l] =simple_comp(Result_fit_HSENS2)
	#prop_comp_rot[l] =simple_comp(Result_fit_rot)
	P_HSENS1.append(Result_peak_HSENS1)
	P_HSENS2.append(Result_peak_HSENS2)
	P_HSENS3.append(Result_peak_HSENS3)
	P_HSENS4.append(Result_peak_HSENS4)
	P_ROT.append(Result_peak_rot)



np.sum(P_ROT[-1][M_ROT[-1]>t1,0]>1)
np.sum(P_ROT[-1][M_ROT[-1]>t1,1]==1)
np.sum(P_ROT[-1][M_ROT[-1]>t1,2]>0.5)

Hue = np.arange(0,2*np.pi,2*np.pi/24)
y = MM1[-1][29,-1,:]
plt.plot(Hue*180/np.pi,y,linewidth = 1,color = 'blue')
#plt.plot(Hue*180/np.pi,yhat, color='red')
plt.show()

F.peak_detection(y, show = True)



'''
fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
#plt.title('Layer %s' %str(i+1))
for i in range(0,5):
	h = ax[np.unravel_index(i,(2,3))].hist(P_ROT[i][:,0],color ='#343837')
	#ax[np.unravel_index(i,(2,2))].set_ylim(0,np.amax(h[0]))
	ax[np.unravel_index(i,(2,3))].set_title('Layer %s'%str(i+1),fontsize=25)
	plt.setp(ax[np.unravel_index(i,(2,3))].get_xticklabels(),fontsize=20)
	plt.setp(ax[np.unravel_index(i,(2,3))].get_yticklabels(),fontsize=20)

fig.text(0.5, 0.01, 'Number peaks', ha='center',fontsize = 25)
fig.text(0.04, 0.5, 'Number kernels', va='center', rotation='vertical', fontsize = 25)

fig.tight_layout()
plt.show(fig)
plt.close(fig)'''


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()




prop_null = np.zeros(len(CLASS))
prop_1 = np.zeros(len(CLASS))
prop_2 = np.zeros(len(CLASS))
prop_3 = np.zeros(len(CLASS))
prop_4 = np.zeros(len(CLASS))
for l in range(0,len(CLASS)):
	a1 = np.amax(M_HSENS1[l],axis = -1)<t2
	a2 = np.amax(M_HSENS2[l],axis = -1)<t2
	a3 = np.amax(M_HSENS3[l],axis = -1)<t2
	a4 = np.amax(M_HSENS4[l],axis = -1)<t2
	ar = np.amax(M_ROT[l],axis = -1)<t2
	conc = np.concatenate((np.amax(P_HSENS1[l], axis = 1)[a1,0], np.amax(P_HSENS2[l], axis = 1)[a2,0], np.amax(P_HSENS3[l], axis = 1)[a3,0], np.amax(P_HSENS4[l], axis = 1)[a4,0], np.amax(P_ROT[l], axis = 1)[ar,0]))
	prop_null[l] = np.sum(conc == 0).astype(float)/conc.size
	prop_1[l] = np.sum(conc == 1).astype(float)/conc.size
	prop_2[l] = np.sum(conc == 2).astype(float)/conc.size
	prop_3[l] = np.sum(conc == 3).astype(float)/conc.size
	prop_4[l] = np.sum(conc == 4).astype(float)/conc.size
 

### FIGURE
fig = plt.figure(figsize = (7,7))
'''
left, bottom, width, height = [0.12, 0.31, 0.41, 0.59]
rect_ax1 = [left, bottom, width, height]


ax1 = plt.axes(rect_ax1)

ax1.plot(np.arange(1,len(CLASS)+1),(prop_1+prop_2+prop_3+prop_4)*100,'k',linewidth = 1)
ax1.fill_between(np.arange(1,len(CLASS)+1),100,(prop_1+prop_2+prop_3+prop_4)*100,color = [1,1,1],label = 'flat')
ax1.plot(np.arange(1,len(CLASS)+1),(prop_2+prop_3+prop_4)*100,'k',linewidth = 1)
ax1.fill_between(np.arange(1,len(CLASS)+1),(prop_1+prop_2+prop_3+prop_4)*100,(prop_2+prop_3+prop_4)*100,color = [0.7,0.7,0.7],label = '1 peak')
ax1.plot(np.arange(1,len(CLASS)+1),(prop_3+prop_4)*100,'k',linewidth = 1)
ax1.fill_between(np.arange(1,len(CLASS)+1),(prop_2+prop_3+prop_4)*100,(prop_3+prop_4)*100,color = [0.6,0.60,0.60],label = '2 peaks')
ax1.plot(np.arange(1,len(CLASS)+1),prop_4*100,'k',linewidth = 1)
ax1.fill_between(np.arange(1,len(CLASS)+1),(prop_3+prop_4)*100,prop_4*100,color = [0.4,0.4,0.4],label = '3 peaks')

ax1.fill_between(np.arange(1,len(CLASS)+1),prop_4*100,0,color = [0.2,0.2,0.2],label = '4 peaks')



ax1.set_ylim(0,100,emit=True)
ax1.set_xticks(np.arange(1,len(CLASS)+1, len(CLASS)//5))
ax1.set_title('Non hue selective',fontsize=17)
#plt.setp(ax2.get_xticklabels(), fontsize=15)
#plt.setp(ax2.get_yticklabels(), fontsize=15)

'''


#rect_ax2 = [left+width+0.02, bottom, width, height]
rect_ax2 = [0.12, 0.22, 0.84, 0.74]
prop_null = np.zeros(len(CLASS))
prop_1 = np.zeros(len(CLASS))
prop_2 = np.zeros(len(CLASS))
prop_3 = np.zeros(len(CLASS))
prop_4 = np.zeros(len(CLASS))
for l in range(0,len(CLASS)):
	a1 = np.amax(M_HSENS1[l],axis = -1)>t2
	a2 = np.amax(M_HSENS2[l],axis = -1)>t2
	a3 = np.amax(M_HSENS3[l],axis = -1)>t2
	a4 = np.amax(M_HSENS4[l],axis = -1)>t2
	ar = np.amax(M_ROT[l],axis = -1)>t2
	#conc = np.concatenate((np.amax(P_HSENS1[l], axis = 1)[a1,0], np.amax(P_HSENS2[l], axis = 1)[a2,0], np.amax(P_HSENS3[l], axis = 1)[a3,0], np.amax(P_HSENS4[l], axis = 1)[a4,0], np.amax(P_ROT[l], axis = 1)[ar,0]))
	conc = np.concatenate((np.amax(P_HSENS1[l], axis = 1)[a1,0], np.amax(P_HSENS2[l], axis = 1)[a2,0], np.amax(P_HSENS3[l], axis = 1)[a3,0], np.amax(P_HSENS4[l], axis = 1)[a4,0]))
	#conc = np.concatenate((P_HSENS1[l], axis = 1)[a1,0], np.amax(P_HSENS2[l], axis = 1)[a2,0], np.amax(P_HSENS3[l], axis = 1)[a3,0], np.amax(P_HSENS4[l], axis = 1)[a4,0]))
	prop_null[l] = np.sum(conc == 0).astype(float)/conc.size
	prop_1[l] = np.sum(conc == 1).astype(float)/conc.size
	prop_2[l] = np.sum(conc == 2).astype(float)/conc.size
	prop_3[l] = np.sum(conc == 3).astype(float)/conc.size
	prop_4[l] = np.sum(conc == 4).astype(float)/conc.size
 
#ax2 = plt.axes(rect_ax2)
ax2 = plt.axes(rect_ax2)
ax2.plot(np.arange(1,len(CLASS)+1),(prop_1+prop_2+prop_3+prop_4)*100,'k',linewidth = 1)
#ax2.fill_between(np.arange(1,len(CLASS)+1),100,(prop_1+prop_2+prop_3+prop_4)*100,color = [1,1,1],label = '1 peak')
ax2.plot(np.arange(1,len(CLASS)+1),(prop_2+prop_3+prop_4)*100,'k',linewidth = 1)
ax2.fill_between(np.arange(1,len(CLASS)+1),(prop_1+prop_2+prop_3+prop_4)*100,(prop_2+prop_3+prop_4)*100,color = [1,1,1],label = '1 peak')
ax2.plot(np.arange(1,len(CLASS)+1),(prop_3+prop_4)*100,'k',linewidth = 1)
ax2.fill_between(np.arange(1,len(CLASS)+1),(prop_2+prop_3+prop_4)*100,(prop_3+prop_4)*100,color = [0.7,0.70,0.70],label = '2 peaks')
#ax2.plot(np.arange(1,len(CLASS)+1),prop_4*100,'k',linewidth = 1)
ax2.fill_between(np.arange(1,len(CLASS)+1),(prop_3+prop_4)*100,0,color = [0.4,0.4,0.4],label = '3 peaks')

#ax2.fill_between(np.arange(1,len(CLASS)+1),prop_4*100,0,color = [0.2,0.2,0.2],label = '4 peaks')

#plt.setp(ax2.get_xticklabels(), fontsize=15)
#plt.setp(ax2.get_yticklabels(), fontsize=15)

ax2.set_ylim(0,100,emit=True)
ax2.set_xlim(1,len(CLASS),emit=True)
ax2.set_ylabel('% of kernels')
ax2.set_xlabel('Layer')
#ax2.set_yticks([])
ax2.set_xticks(np.arange(1,len(CLASS)+1, len(CLASS)//5))

#ax2.legend(loc='upper center', fontsize = 14)
#ax2.legend(bbox_to_anchor=(-1.04, 0, 2.04, -0.24), loc=1,
#           ncol=3, mode="expand", borderaxespad=0.,fontsize=14)
ax2.legend(bbox_to_anchor=(-0.04, 0, 1.04, -0.15), loc=1,
           ncol=3, mode="expand", borderaxespad=0.,fontsize=14)

#fig.text(0.02, 0.57, '% of kernels', va='center', rotation='vertical', fontsize = 17)
#fig.text(0.53, 0.195, 'Layer', ha='center',fontsize = 17)
#ax2.set_title('Hue selective',fontsize=17)

#fig.tight_layout()
plt.show()
plt.close()
fig.savefig('../figures/nb_auxiliary_peaks_' + name_net, dpi = 300)



# In[9]:
PEAK_HSENSALL = list()
DIST_ARG_MAX_HSENSALL = list()

for l in range(len(CLASS)):
	peak_HSENSALL = list()
	dist_arg_max_HSENSall = np.array([])
	for k in range(len(CLASS[l])):
		dist_arg_max_HSENSall = F.dist_arg_max_loop(dist_arg_max_HSENSall,ARG_HSENS1[l][k,-1],MM1[l][k,-1])
		dist_arg_max_HSENSall = F.dist_arg_max_loop(dist_arg_max_HSENSall,ARG_HSENS2[l][k,-1],MM2[l][k,-1])
		dist_arg_max_HSENSall = F.dist_arg_max_loop(dist_arg_max_HSENSall,ARG_HSENS3[l][k,-1],MM3[l][k,-1])
		dist_arg_max_HSENSall = F.dist_arg_max_loop(dist_arg_max_HSENSall,ARG_HSENS4[l][k,-1],MM4[l][k,-1])
	dist_arg_max_HSENSall = dist_arg_max_HSENSall[~(dist_arg_max_HSENSall==0)] # Select only secondary peaks
	dist_arg_max_HSENSall = F.conversion_dist_arg(dist_arg_max_HSENSall*15,0)
	DIST_ARG_MAX_HSENSALL.append(dist_arg_max_HSENSall.copy())
	PEAK_HSENSALL.append(peak_HSENSALL)




D.DEFINE_PLT_RC()

fig = plt.figure()
ax = fig.add_subplot(111)
h = np.histogram(DIST_ARG_MAX_HSENSALL[-1],bins = np.arange(0,181,15))
ax.bar(h[1][:-1],h[0].astype(float)/DIST_ARG_MAX_HSENSALL[-1].size,width = h[1][1] -h[1][0], color ='#343837')
plt.xlabel('Distance from preferred hue (degrees)')
plt.ylabel('Frequency')

fig.tight_layout()
plt.show()
fig.savefig('../figures/distance_inter_peak_' + name_net,dpi = 300)

plt.close(fig)

# In[9]:


D.DEFINE_PLT_RC()

### ---------------------------------------------------------------------------------------------------------------------------------
# Plot tuning curves GDR
#

K = 16
Y = np.array([MM1[-1][K,:], MM2[-1][K,:], MM3[-1][K,:], MM4[-1][K,:]])

fig, ax = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=True)
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        p = ax[i,j].plot(Hue*180/np.pi,Y[i][j,:], 'k')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
	#ax[np.unravel_index(i,(2,2))].set_ylim(0,np.amax(h[0]))
#ax.plot(Hue*180/np.pi,y.T[:,:],'--k',linewidth = 4,label = 'Kernel response')
#ax.plot(Hue*180/np.pi,y.T[:,-1],'k',linewidth = 4,label = 'Kernel response')
#ax.plot(Hue*180/np.pi,y.T[:,1],'b',linewidth = 4,label = 'Kernel response')
#ax.axhline(GREY[-1][K], xmin=0, xmax=345, color='k', ls='--',linewidth = 6, label='Response to grey')
#ax.axhline(ROT[-1][K,0], xmin=0, xmax=345,color='r',ls='--',linewidth = 6,label = 'Response to original')
#plt.xlabel('Hue angle',fontsize = 25)
#plt.ylabel('Kernel response',fontsize = 25)
#plt.ylim(0,250)
#ax.set_xticks([])
#ax.set_yticks([])
#plt.xlim(0.5,5.5)
#plt.legend(loc='lower right', fontsize = 20)
fig.tight_layout()
plt.show()
fig.savefig('../figures/set_tuning_curves',dpi = 300)
plt.close(fig)



# In[9]: Some global values


Total_nb_kernels = 0
for i in range(len(CLASS)):
    Total_nb_kernels = Total_nb_kernels + len(CLASS[l])

print('Total number of kernels within net = ')
print(Total_nb_kernels)
