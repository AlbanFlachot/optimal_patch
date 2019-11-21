#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:42:02 2019

@author: alban
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import pickle





def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')





def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)
    #print ind
    return ind


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



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi])

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan(z/np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi
    return np.array([r, az, elev])

## Rotation matrix along the x axis (luminance axis)
def rotation(X,teta):
	RM = np.array([[1,0,0],[0,np.cos(teta),-np.sin(teta)],[0,np.sin(teta),np.cos(teta)]])
	return np.dot(X,RM.T)


### Function that converts RGB to Opponent Space ( Plataniotis and A. Venetsanopoulos.Color Image Pro-cessing and Application. Springer, 2000).
def RGB2PCA(x):
	#M = np.array([[ 0.56677561,  0.71836896,  -0.40189701],[ 0.58101187,  0.02624069,  0.81321349],[ 0.58406993,  -0.69441022,  -0.41888652]])
	M = np.array([[ 0.66666,  1,  -0.5],[ 0.66666,  0,  1],[ 0.66666,  -1,  -0.5]])
	return np.dot(x,M)

def PCA2RGB(x):
	#M = np.array([[ 0.56677561,  0.71836896,  -0.40189701],[ 0.58101187,  0.02624069,  0.81321349],[ 0.58406993,  -0.69441022,  -0.41888652]])
	M = np.array([[ 0.66666,  1,  -0.5],[ 0.66666,  0,  1],[ 0.66666,  -1,  -0.5]])
	return np.dot(x,np.linalg.inv(M))



    
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



def DISTRIB_resp(RESP,Treshs, resp_type = 1):
	DIS_R = np.zeros((len(RESP),len(Treshs)))
	if resp_type == 1:
        	count = 0
        	for l in range(0,len(RESP)):
        		#M = np.amax(np.stack((M_HSENS1[l],M_HSENS2[l],M_HSENS3[l],M_HSENS4[l]),axis = 1),axis = -1)
        		#MEAN_M[l] = np.mean(M)
        		DIS_R[l,0] = respo(RESP[l],Treshs[0])*100
        		DIS_R[l,1] = respo(RESP[l],Treshs[1])*100
        		DIS_R[l,2] = respo(RESP[l],Treshs[2])*100
        		DIS_R[l,3] = respo(RESP[l],Treshs[3])*100
        		DIS_R[l,4] = respo(RESP[l],Treshs[4])*100
        		DIS_R[l,5] = respo(RESP[l],Treshs[5])*100
        		DIS_R[l,6] = respo(RESP[l],Treshs[6])*100
        		DIS_R[l,7] = respo(RESP[l],Treshs[7])*100
        		count +=1

	else:
        	count = 0
        	for l in range(0,len(RESP)):
        		#M = np.amax(np.stack((M_HSENS1[l],M_HSENS2[l],M_HSENS3[l],M_HSENS4[l]),axis = 1),axis = -1)
        		#MEAN_M[l] = np.mean(M)
        		DIS_R[l,0] = respo2(RESP[l],Treshs[0])*100
        		DIS_R[l,1] = respo2(RESP[l],Treshs[1])*100
        		DIS_R[l,2] = respo2(RESP[l],Treshs[2])*100
        		DIS_R[l,3] = respo2(RESP[l],Treshs[3])*100
        		DIS_R[l,4] = respo2(RESP[l],Treshs[4])*100
        		DIS_R[l,5] = respo2(RESP[l],Treshs[5])*100
        		DIS_R[l,6] = respo2(RESP[l],Treshs[6])*100
        		DIS_R[l,7] = respo2(RESP[l],Treshs[7])*100
        		count +=1
	return DIS_R


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


def misclass(CLASS,M,t2,i):
	Class_HSENS1 = list()
	C_HSENS1 = np.zeros(len(CLASS))
	C_HSENS1_tot = np.zeros(len(CLASS))
	A_WRONG= np.zeros(len(CLASS))
	A_RIGHT= np.zeros(len(CLASS))
	for l in range(0,len(CLASS)):
		A_wrong = np.zeros(CLASS[l].shape[:-2])
		A_right = np.zeros(CLASS[l].shape[:-2])
		for k in range(0,len(CLASS[l])):
			A_wrong[k] = (0 in CLASS[l][k,:,:,i])*1
			A_right[k] = 1 - (0 in CLASS[l][k,:,:,i])*1

		A_WRONG[l]= np.sum(A_wrong[(M[l]>t2)])/A_wrong.size
		A_RIGHT[l]= np.sum(A_right[(M[l]>t2)])/A_right.size
		Class_HSENS1.append(A_wrong)
		C_HSENS1[l] = np.sum(Class_HSENS1[l][M[l]>t2])/(Class_HSENS1[l][M[l]>t2]).size
		C_HSENS1_tot[l] = np.sum(Class_HSENS1[l])/(Class_HSENS1[l]).size

	return Class_HSENS1, C_HSENS1, C_HSENS1_tot


def prop_misclass(CLASS, M_HSENS1, t2, i):
	prop_A = np.zeros(len(CLASS))
	prop_non_A = np.zeros(len(CLASS))
	for l in range(0,len(CLASS)):
		C_R = CLASS[l][:,-1,0,-1]
		A_wrong = np.zeros(len(CLASS[l]))
		for k in range(0,len(CLASS[l])):
			A_wrong[k] = (0 in CLASS[l][k,1:,:,i])*1
		#print np.sum(C_R)/C_R.size # nb of kernels that were right at the beginning
		#print np.sum(C_R[M_HSENS1[l]>t2])/C_R[M_HSENS1[l]>t2].size # nb of color sele kernels that were right at the beginning
		prop_A[l] = np.sum(A_wrong[(C_R == 1 ) & (np.amax(M_HSENS1[l], axis = -1)>t2)])/np.sum(C_R[np.amax(M_HSENS1[l], axis = -1)>t2]) ## = proportion of col selective kernels which were right without transformation but wrong after sur transformation
		prop_non_A[l] = np.sum(A_wrong[(C_R == 1 ) & (np.amax(M_HSENS1[l], axis = -1)<t2)])/np.sum(C_R[np.amax(M_HSENS1[l], axis = -1)<t2]) ## = proportion of non col selective kernels which were right without transformation but wrong after obj transformation
		#print ('\n')

	return prop_A, prop_non_A

def prop_misclass_all(CLASS, SENS, t2):
	'''funcion taht computes the proportion of images that were misclassified but originally correctly classified
	INPUT: CLASS: binary array of correct or incorrect classification, SENS arrays of OCS, t2 treshold for sensivity.'''
	prop_A = np.zeros(len(CLASS)) #
	prop_non_A = np.zeros(len(CLASS))
	prop_all = np.zeros(len(CLASS))
	for l in range(0,len(CLASS)):
		C_R = CLASS[l][:,0,0,-1] # array of cases where the original image was correctly classified or not
		A_wrong = np.zeros(len(CLASS[l]))
		for k in range(0,len(CLASS[l])):
			A_wrong[k] = (0 in CLASS[l][k,4,:,:-1])*1 # binary array saying whether the images was misclassified, among all color modifications (rot non included)
		S = np.amax(SENS[l], axis = (-1))
		prop_A[l] = np.sum(C_R[ (A_wrong == 1 ) & (S >t2)])/np.sum(C_R[S >t2]) ## = proportion of col selective kernels which were right without transformation but wrong after sur transformation
		prop_non_A[l] = np.sum(C_R[ (A_wrong == 1 ) & (S <t2) ]) / np.sum(C_R[S <t2]) ## = proportion of non col selective kernels which were right without transformation but wrong after obj transformation
		prop_all[l] = np.sum(C_R[ (A_wrong == 1 ) ]) / np.sum(C_R) ## = proportion of non col selective kernels which were right without transformation but wrong after obj transformation
		

	return prop_A, prop_non_A, prop_all


def peak_detection(y,show = False):
	'''Function which detects te number of peaks in a curve, according to a certain threshold.
	   First peak must be 1/8 of the max and second peak must be 1/3 or more of the main peak'''
	#print (np.amax(y) - np.amin(y))
	#yhat = savitzky_golay(np.concatenate((y[-2:],y)), 5, 2) # window size 51, polynomial order 3
	if (np.amax(y) - np.amin(y)) > np.amax(y)/2: # if the curve shows a variation of at least 1/4, we can conduct the analysis.
		ydet = np.concatenate((y[-2:],y))
		p = detect_peaks( ydet, mph=(np.amax(y)-np.amin(y))/2 + np.amin(y), mpd=4, threshold=0, edge='rising',kpsh=False, valley=False, show=show, ax=None)
		#return p,len(p)
		if len(p) <2: # if the tuning curve has 1 peak only or less
			return np.array([float(len(p)),False,False])
		else:
			Boole = 0
			sec_peak = 0
			for i in range(len(p)):
				if (10 < np.absolute(p[i] - np.argmax(ydet)) < 14): #if there is at least a secondary peak, find whether it 
					Boole = 1
					sec_peak = ((ydet[p[i]]-np.amin(ydet))/ (np.amax(ydet)-np.amin(ydet)))

			return np.array([float(len(p)),Boole,sec_peak])
	else:
		#print ('no variation')
		return 0,0,0 

def peak_detection2(y):
	'''Function which detects te number of peaks in a curve, according to a certain threshold.
	   First peak must be 1/8 of the max and second peak must be 1/3 or more of the main peak'''
	#print (np.amax(y) - np.amin(y))
	#yhat = savitzky_golay(np.concatenate((y[-2:],y)), 5, 2) # window size 51, polynomial order 3
	if (np.amax(y) - np.amin(y)) > np.amax(y)/8: # if the curve shows a variation of at least 1/8, we can conduct the analysis.
		ydet = np.concatenate((y[-2:],y))
		p = detect_peaks( ydet, mph=(np.amax(y)-np.amin(y))/3 + np.amin(y), mpd=4, threshold=0, edge='rising',kpsh=False, valley=False, show=False, ax=None)
		#return p,len(p)
		return (p-2)%24
	else:
		return np.nan

  
 
  
def conversion_dist_arg(dist_arg,radian):
	if ~radian:
		dist_arg = dist_arg*np.pi/180

	dist = np.arccos(np.cos(dist_arg))
	#dist = dist_arg
	if ~radian:
		dist = dist*180/np.pi

	return dist

def dist_arg_max(p,argmax):
	if np.isnan(p).any():
		return np.array([0])
	else:
		dist = np.absolute(p - argmax)
		dist = conversion_dist_arg(dist*15,0)/15
		dist = np.sort(dist)
		dist[0] = 0 # Put the peak found the closest to argmax as the 0 dist (to nullify some errors)
		return dist

def dist_arg_max_loop(dist_arg_max_HSENSall,ARG_HSENS,y):
	p = peak_detection2(y)
	return np.concatenate((dist_arg_max_HSENSall,dist_arg_max(p,ARG_HSENS/15)))

