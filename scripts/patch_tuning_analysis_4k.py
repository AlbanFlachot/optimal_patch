from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import pickle


import sys
sys.path.append('../')

from utils import FUNCTIONS as F # script with a bunch of functions
from utils import DISPLAYS as D # script with functions to display

caffe_root = '/home/alban/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
#sys.path.insert(0, caffe_root + 'python')

# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')

	## Fit functions
	#____________________________________________________________________________________________________________________

    
# In[9]:
###__________________________________________________________________________________________________________________________________

import os
if not os.path.exists('../figures'):
		os.mkdir('../figures')

name_net = 'VGG-19'

print('We start analysis for net: ' + name_net)

'''
Data obtained for local color modifications of segments

MAX is the maximal response of the kernels
GREY the responso of the kernels to the segments in black and white
MM* are the kernels' activations to color manipulations within segments *
CLASS are the class given by the models for all stimuli.
Good_Class gives the correct class for all stimuli
'''
r = open('../pickles/' + name_net + '_patches_4k_Sat.pickle','rb') 
MAX, GREY,MM1,MM2,MM3,MM4,ROT,CLASS,Good_Class= pickle.load(r, encoding="bytes")
r.close()


'''
Data obtained for global color modifications of entire images

Same as before without MAX and for the whole image modif, CLASS_GREY is the classification of the model for B&W stimuli
'''
r = open('../pickles/' + name_net + '_patches_GREY_ROT.pickle','rb')
GREY_whole, ROT_whole, CLASS_whole, CLASS_GREY, Good_Class_whole = pickle.load(r, encoding="bytes")
r.close()

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

ARG_HSENS1 = list() # indexes of highest response == preferred hues
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
	MM1[l][np.amax(MM1[l][:],axis = -1)==0] = 1.0 #There are a few instances where at 0 chroma, the kernel is absolutely non responsive. In which case the computation of hue selectivity is 0/0 and generates nan value. To avoid subsequent issues we put these rare cases at 1.
	MM2[l][np.amax(MM2[l][:],axis = -1)==0] = 1.0
	MM3[l][np.amax(MM3[l][:],axis = -1)==0] = 1.0
	MM4[l][np.amax(MM4[l][:],axis = -1)==0] = 1.0
	T = np.amax((MM1[l],MM2[l],MM3[l],MM4[l]),axis = (0,-1)) #max across partial kernels and hue
	Tpk = np.amax((MM1[l],MM2[l],MM3[l],MM4[l]),axis = -1) # max across hue
	P = np.amin((MM1[l],MM2[l],MM3[l],MM4[l]),axis = (0,-1)) # min across kernels and hue
	BIG_MAX = np.amax((np.amax(T,axis = 1),np.amax(ROT_whole[l],axis = 1),GREY[l] ),axis =0)
	SENSITIVITY.append((T-P)/T)
	RESP.append(((T.T-GREY[l]).T)/T)
	C2A.append((T.T-GREY[l]).T/(T.T+GREY[l]).T)
	M_HSENS1.append((np.amax(MM1[l],axis = -1)-np.amin(MM1[l],axis = -1))/np.amax(MM1[l],axis = -1))
	M_HSENS2.append((np.amax(MM2[l],axis = -1)-np.amin(MM2[l],axis = -1))/np.amax(MM2[l],axis = -1))
	M_HSENS3.append((np.amax(MM3[l],axis = -1)-np.amin(MM3[l],axis = -1))/np.amax(MM3[l],axis = -1))
	M_HSENS4.append((np.amax(MM4[l],axis = -1)-np.amin(MM4[l],axis = -1))/np.amax(MM4[l],axis = -1))
	M_ROT.append((np.amax(ROT[l],axis = -1)-np.amin(ROT[l],axis = -1))/np.amax(ROT[l],axis = -1))
	hsensall = np.array([M_HSENS1[l],M_HSENS1[l],M_HSENS1[l],M_HSENS1[l]])
	hsensall = np.moveaxis(hsensall,0,-1)
	M_HSENSALL.append(hsensall)
	SensSAT = 1 - np.amin(Tpk[:,:,:],axis = -1)/np.amax(Tpk[:,:,:],axis = -1) # change in max response with saturation
	RespSAT = 1 - np.amin(T,axis = -1)/np.amax(T,axis = -1) # Saturation responsivity
	ARG_HSENS1.append(np.argmax(MM1[l],axis = -1)*15)
	ARG_HSENS2.append(np.argmax(MM2[l],axis = -1)*15)
	ARG_HSENS3.append(np.argmax(MM3[l],axis = -1)*15)
	ARG_HSENS4.append(np.argmax(MM4[l],axis = -1)*15)
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





	# Plot distribution Responsivity
Treshs = np.array([0,1/8,2/8,3/8,4/8,5/8,6/8,7/8])
#D.PLOT_FIGURE_GRADUATE_DISTRIB(RESP,Treshs,'CR')
DIS_CR = F.DISTRIB_resp(RESP,Treshs)



# In[9]:


# plot global sensitivity proportions
DIS_OCS = F.DISTRIB_resp(SENSITIVITY,Treshs, resp_type = 1)


print('Proportion of weakly color sensitive kernels in first layer is %f' %(DIS_OCS[0,0]-DIS_OCS[0,1]))
print('Proportion of highly color sensitive kernels in first layer is %f' %(DIS_OCS[0,-1]))

#D.PLOT_FIGURE_GRADUATE_DISTRIB(SENSITIVITY,Treshs,' OCS', resp_type = 1)

SENSt1,SENSt2,SENSt3,Mean_SENS,Std_SENS = F.RESPO(SENSITIVITY,t1,t2,t3)



#-------------------------------------------------------------------------------------------------------------
#### CS_HSENS1

selecMM1t1,selecMM1t2,selecMM1t3,Mean_selec_HSENS1,Std_selec_HSENS1 = F.RESPO(M_HSENS1,t1,t2,t3)



#-------------------------------------------------------------------------------------------------------------
#### CS_HSENS2
selecMM2t1,selecMM2t2,selecMM2t3,Mean_selec_HSENS2,Std_selec_HSENS2 = F.RESPO(M_HSENS2,t1,t2,t3)


#-------------------------------------------------------------------------------------------------------------
#### CS_HSENS3

selecMM3t1,selecMM3t2,selecMM3t3,Mean_selec_HSENS3,Std_selec_HSENS3 = F.RESPO(M_HSENS3,t1,t2,t3)



#-------------------------------------------------------------------------------------------------------------
#### CS_HSENS4
selecMM4t1,selecMM4t2,selecMM4t3,Mean_selec_HSENS4,Std_selec_HSENS4 = F.RESPO(M_HSENS4,t1,t2,t3)



#-------------------------------------------------------------------------------------------------------------
#### CS_rot


selecrott1,selecrott2,selecrott3,Mean_selec_rot,Std_selec_rot = F.RESPO(M_ROT,t1,t2,t3)

#PLOT_FIGURE('CSROT',selecrott1,selecrott2,selecrott3,Mean_selec_rot)



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





# In[9]: Proportion of saturation sensitive kernels

DIS_CCR = F.DISTRIB_resp(SENS_SAT,Treshs, resp_type = 0)



senssatt1,senssatt2,senssatt3,Mean_senssat,Std_senssat = F.RESPO(SENS_SAT,t1,t2,t3, resp_type = 0)



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
#### Preferred hues

'''
Part of the script where we compute the preferred hues of our models. The tricky part is in the case of segments of the same kernel selective to the same hue(+-30°): we then consider the kernel selective for only one hue, the mean.  

'''

import itertools # Library for smart generators. Allow combinations of elements for instance

def mean_angle(angle_array):
	'''
	Functions that computes the angular mean of several angles, within the input array.
	Input:
		angle_array: 
	'''
	meansin1 = np.arcsin(np.mean(np.sin(angle_array*np.pi/180)))*180/np.pi
	meancos1 = np.arccos(np.mean(np.cos(angle_array*np.pi/180)))*180/np.pi
	return meancos1*np.sign(meansin1)


ARG_SEL = list() # List of preferred hues
Nb_col_select = list() # list of the number of preferred hues per kernel
for l in range(len(M_HSENS1)):
	nb_kernels = len(M_HSENS1[l]) # nb of kernels
	P_sel = np.array([np.amax(M_HSENS1[l], axis = -1) > t2, 
						np.amax(M_HSENS2[l], axis = -1) > t2, 
						np.amax(M_HSENS3[l], axis = -1) > t2, 
						np.amax(M_HSENS4[l], axis = -1) > t2]) # array of hue selective segments [ nb_segments, nb_kernels]
	Nb_col_select.append( np.sum(P_sel,axis = 0)) # nb of hue selective segments per kernels [nb_layers][ nb_kernels]
	PREF_HUES = np.array([ARG_HSENS1[l][range(len(ARG_HSENS1[l])),np.argmax(M_HSENS1[l], axis = -1)], 
								ARG_HSENS2[l][range(len(ARG_HSENS2[l])),np.argmax(M_HSENS2[l], axis = -1)], 
								ARG_HSENS3[l][range(len(ARG_HSENS3[l])),np.argmax(M_HSENS3[l], axis = -1)], 
								ARG_HSENS4[l][range(len(ARG_HSENS4[l])),np.argmax(M_HSENS4[l], axis = -1)]]) # array of preferred hues, all hue selectivities [nb segments, nb kernels]
	Arg_sel = list()  # List of preferred hues for this layer
	for idx in np.where(Nb_col_select[l] > 0)[0]: # in the case of color slective kernels
		pref_hues = PREF_HUES[P_sel[:,idx], idx] # pref hues found for kernel ''idx'' on hue selective segments
		print(pref_hues)
		if len(pref_hues) > 1: # if more than one segment is hue selective
			pairs = np.array([i for i in itertools.combinations(pref_hues,2)]) # compute all possible combinations of pairs
			diffs = [np.arccos(np.cos((i[0] -i[1])*np.pi/180))*180/np.pi for i in pairs] # angle diffs for each pair
			crit = np.array(diffs)<45 # boolean array if angular diff is bellow a certain threshold, in which case we consider them similar and take the mean
			if True in crit: 
				good_hues = np.array([]) # define a new array with the correct preferred hues
				for i in pref_hues:
					if i not in pairs[crit]: # if a hue is not a doublon, save it in good_hues
						good_hues = np.concatenate((good_hues,np.array([i])))
				samehue1 = np.array([])
				samehue2 = np.array([]) # up to 2 possible pairs with similar hues (4 segments)
				for i in np.where(crit)[0]: # where there is a doublon
					if len(samehue1) == 0: # if its the first step
						samehue1 = pairs[i] # we save the pair as the same hue
					else:
						if True in (((samehue1 - pairs[i][0])<45) | ((samehue1 - pairs[i][1])<45)):
							samehue1 = np.concatenate((samehue1,pairs[i])) # if it is not the first step and at least one of the new doublon's element is equal to the saved one, then we save it as being the same hue
						else:
							if len(samehue2) == 0: # otherwise we save it as a new hue
								samehue2 = pairs[i]
							else:
								if True in (((samehue2 - pairs[i][0])<45) | ((samehue2 - pairs[i][1])<45)):
									samehue2 = np.concatenate((samehue2,pairs[i]))
						
				hue1 = mean_angle(samehue1) # we compute the mean of the cluster of the first redundant hue
				good_hues = np.concatenate((good_hues,np.array([hue1]))) # we save this mean as good hue
				if len(samehue2) >0: # if there is a second redundant hue, we compute the mean
					#print((l,samehue2))
					hue2 = np.array([mean_angle(samehue2)])
					good_hues = np.concatenate((good_hues, hue2)) # and save it as good hue
				pref_hues = good_hues
			Nb_col_select[l][idx] = len(pref_hues) # we then rectify the number of hues
		
		Arg_sel.append(pref_hues) # update the list of preferred hues
	ARG_SEL.append(Arg_sel)



nb_lay = len(M_HSENS1)
Theta = np.arange(0,2*np.pi,2*np.pi/nb_lay)
Sat = 0.43
Lum = 0.2

(x,y) = F.pol2cart(Sat, Theta)

color_id = F.PCA2RGB(np.array([[np.zeros(nb_lay)+Lum],[x],[y]]).T)+0.5
#color_id = PCA2RGB(np.array([[Lum],[np.zeros(nb_lay)],[np.zeros(nb_lay)]]).T)+0.55
color_id = color_id.reshape((len(color_id),3))




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



print('Performance drops by %f percents after hue modification 0.25 chroma' %(( np.mean(Classification[-1,1,:,:-1], axis = -1).max() - np.mean(Classification[-1,1,:,:-1], axis = -1).min() )/np.mean(Classification[-1,1,:,:-1], axis = -1).max()))
print('Performance drops by %f percents after hue modification 1 chroma' %(( np.mean(Classification[-1,-1,:,:-1], axis = -1).max() - np.mean(Classification[-1,-1,:,:-1], axis = -1).min() )/np.mean(Classification[-1,-1,:,:-1], axis = -1).max()))


# Figure rotation whole image

WEIGHTS_CLASSIFICTION = [len(CLASS[l]) for l in range(len(CLASS))]

MEAN_CLASSIFICATION_ROT = np.average(CLASSIFICATION_ROT,weights = WEIGHTS_CLASSIFICTION, axis = 0)

CLASSIFICATION_GREY = [np.sum(CLASS_GREY[l])*100/CLASS_GREY[l].size for l in range(len(CLASS))]

MEAN_CLASSIFICATION_GREY = np.average(CLASSIFICATION_GREY,weights = WEIGHTS_CLASSIFICTION)



print('Performance drops by %f percents after conversion to B&W' %((MEAN_CLASSIFICATION_ROT[0]*100-MEAN_CLASSIFICATION_GREY)/MEAN_CLASSIFICATION_ROT[0]))
print('Performance drops by %f percents after hue modification' %((MEAN_CLASSIFICATION_ROT.max()-MEAN_CLASSIFICATION_ROT.min())/MEAN_CLASSIFICATION_ROT.max()))


# In[9]:


### ---------------------------------------------------------------------------------------------------------------------------------
# Peaks detection algorithm
#


from scipy.signal import find_peaks as fp



def peak_detection(y,show = False):
	'''Function which detects te number of peaks in a curve, according to a certain threshold.
	   First peak must be 1/2 of the max and second peak must be 1/2 or more of the main peak, with a 1/4 minimum prominence'''
	ydet = np.concatenate((y[-12:],y,y[:12])) # we repeat the curve a little bit such that peaks at borders are not neglected by algo
	p, prop = fp( ydet, height=0, distance=4, prominence = (np.amax(y)/6,y.max()))
	idx = ((p-12 > -1) & (p-12 < 24))
	p = (p - 12)[idx]
	for key in prop.keys():
		prop[key] = prop[key][idx]
	return p, prop

'''
Loop over all tuning curves to extract the peaks, their coordinates and properties
'''

P_HSENS1 = list()
P_HSENS2 = list()
P_HSENS3 = list()
P_HSENS4 = list()
P_ROT = list()

for l in range(0,len(CLASS)):
	Result_peak_HSENS1 = np.empty(CLASS[l].shape[:2]+tuple([2]),dtype = object)
	Result_peak_HSENS2 = np.zeros(CLASS[l].shape[:2]+tuple([2]),dtype = object)
	Result_peak_HSENS3 = np.zeros(CLASS[l].shape[:2]+tuple([2]),dtype = object)
	Result_peak_HSENS4 = np.zeros(CLASS[l].shape[:2]+tuple([2]),dtype = object)
	Result_peak_rot = np.zeros(CLASS[l].shape[:2]+tuple([2]),dtype = object)
	for K in range(len(CLASS[l])):
		for sat in range(CLASS[l].shape[1]):
			y = MM1[l][K,sat,:]
			Result_peak_HSENS1[K,sat] = peak_detection(y)
			y = MM2[l][K,sat,:]
			Result_peak_HSENS2[K,sat] = peak_detection(y)
			y = MM3[l][K,sat,:]
			Result_peak_HSENS3[K,sat] = peak_detection(y)
			y = MM4[l][K,sat,:]
			Result_peak_HSENS4[K,sat] = peak_detection(y)
			y = ROT[l][K,sat,:]
			Result_peak_rot[K,sat] = peak_detection(y)
	
	P_HSENS1.append(Result_peak_HSENS1)
	P_HSENS2.append(Result_peak_HSENS2)
	P_HSENS3.append(Result_peak_HSENS3)
	P_HSENS4.append(Result_peak_HSENS4)
	P_ROT.append(Result_peak_rot)

#[P_HSENS1,P_HSENS2,P_HSENS3,P_HSENS4] = np.load('result_peak_detection_python3.npy')

NB_peaks1 = list()
NB_peaks2 = list()
NB_peaks3 = list()
NB_peaks4 = list()

for l in range(len(CLASS)):
	nb_peaks1 = np.zeros(CLASS[l].shape[:2])
	nb_peaks2 = np.zeros(CLASS[l].shape[:2])
	nb_peaks3 = np.zeros(CLASS[l].shape[:2])
	nb_peaks4 = np.zeros(CLASS[l].shape[:2])
	for K in range(len(CLASS[l])):
		for sat in range(CLASS[l].shape[1]):
			nb_peaks1[K, sat] = P_HSENS1[l][K, sat,0].shape[0]
			nb_peaks2[K, sat] = P_HSENS2[l][K, sat,0].shape[0]
			nb_peaks3[K, sat] = P_HSENS3[l][K, sat,0].shape[0]
			nb_peaks4[K, sat] = P_HSENS4[l][K, sat,0].shape[0]
	
	NB_peaks1.append(nb_peaks1)
	NB_peaks2.append(nb_peaks2)
	NB_peaks3.append(nb_peaks3)
	NB_peaks4.append(nb_peaks4)

Hue = np.arange(0,2*np.pi,2*np.pi/24)
y = MM1[-1][29,-1,:]
ydet = np.concatenate((y[-2:],y))
fp( ydet, height=(np.amax(y))/2, distance=4, prominence = (np.amax(y)/8,y.max()))

R1 = F.peak_detection(y, show = False)


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
	
	conc = np.concatenate((np.amax(NB_peaks1[l], axis = 1)[a1], np.amax(NB_peaks2[l], axis = 1)[a2], np.amax(NB_peaks3[l], axis = 1)[a3], np.amax(NB_peaks4[l], axis = 1)[a4]))
	
	prop_null[l] = np.sum(conc == 0).astype(float)/conc.size
	prop_1[l] = np.sum(conc == 1).astype(float)/conc.size
	prop_2[l] = np.sum(conc == 2).astype(float)/conc.size
	prop_3[l] = np.sum(conc == 3).astype(float)/conc.size
	prop_4[l] = np.sum(conc == 4).astype(float)/conc.size

np.save('result_peak_detection_python3.npy',[P_HSENS1,P_HSENS2,P_HSENS3,P_HSENS4])


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



