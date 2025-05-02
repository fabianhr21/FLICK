#!/usr/bin/env python
#
# pyQvarsi, Statistics.
#
# Some useful base statistics.
#
# Last rev: 20/01/2021
from __future__ import print_function, division

import numpy as np


def variances(ref,model):
	return np.std(ref)**2, np.std(model)**2

def medians(ref,model):
	return np.median(ref), np.median(model)

def covariance(ref,model,output_matrix=False):
	out = (ref - np.nanmean(ref))*(model - np.nanmean(model))
	return np.nanmean(out) if not output_matrix else out

def R(ref,model,output_matrix=False):
	'''
	Pearson correlation coefficient R
	'''
	return (covariance(ref,model,output_matrix=output_matrix)/(np.nanstd(ref)*np.nanstd(model)))

def R2(ref,model):
	'''
	Coefficient of determination
	'''
	avg = np.nanmean(ref)
	SSres = np.nansum((ref-model)*(ref-model))
	SStot = np.nansum((ref-avg)*(ref-avg))
	return 1 - SSres/SStot

def bias(ref,model):
	'''
	Bias
	ID 4.3
	Calculates the mean error.
	Result of zero does not necessarily indicate low error due to cancellation.
	'''
	return np.nanmean( ref-model )
	
def MSE(ref,model):
	'''
	Mean Square Error
	ID 4.4
	Calculates a mean error (in data units squared), which is not effected by cancellation.
	Squaring the data may cause bias towards large events.
	'''
	return np.nanmean( (ref-model)**2 )

def RMSE(ref,model):
	''' 
	Root mean Square Error
	ID 4.5
	MSE error (4.4) except result is returned in the same units as model,
	which is useful for interpretation.
	'''
	return np.sqrt( MSE(ref,model) )

def MAE(ref,model):
	''' 
	Mean Absolute Error
	ID 4.6
	Similar to RMSE (4.5) except absolute value is used instead. This reduces
	the bias towards large events; however, it also produces a non-smooth operator
	when used in optimisation.
	'''
	return np.nanmean( np.abs(ref-model) )

def RMAE(ref,model):
	''' 
	Root Mean Absolute Error

	Similar to RMSE (4.5) except absolute value is used instead. This reduces
	the bias towards large events; however, it also produces a non-smooth operator
	when used in optimisation.
	'''
	return np.sqrt( MAE(ref,model) )

def AME(ref,model):
	'''
	Absolute Maximum Error
	ID 4.7
	Records the maximum absolute error.
	'''
	return np.nanmax(np.abs(diff(ref,model)))

def RAE(ref,model):
	'''
	Relative Absolute Error
	This compares the total error relative to what the total error
	would be if the mean was used for the model.
	A lower value indicates a better performance, while a score greater
	than one indicates the model is outperformed by using the mean as the prediction.
	'''
	norm = np.nanmean(np.abs(ref - np.nanmean(ref)))
	return MAE(ref,model)/norm

def NSE(ref,model):
	'''
	Coefficient of determination
	Nash - Sutcliff Model Efficiency
	ID 6.1

	This method compares the performance of the model to a model that
	only uses the mean of the observed data.
	A value of 1 would indicate a perfect model, while a value of zero
	indicates performance no better than simply using the mean.
	A negative value indicates even worse performance.
	'''
	return 1.0 - MSE(ref,model)/(ref.std()**2)

def IoA(ref,model):
	'''
	Index Of Agreement
	ID 6.5
	This method compares the sum of squared error to the potential error.
	This method is similar to 6.4 however it is designed to be better at handling
	differences in modelled and observed means and variances.
	Squared differences may add bias to large data value events (Willmott, 1981).
	'''
	array_denom = np.abs(model - np.nanmean(ref)) + np.abs(model + np.nanmean(model))
	denom       = (array_denom**2).sum()
	return 1.0 - len(model)*MSE(ref,model)/denom

def PI(ref,model):
	'''
	Persistence Index
	ID 6.6
	The persistence index compares the sum of squared error to the error
	that would occur if the value was forecast as the previous observed value.
	Similar to 6.1 except the performance of the model is being compared to the previous value.
	'''
	norm = np.sum(np.diff(ref[1:])**2)/len(model)
	return 1.0 - MSE(ref,model)/norm

def RSR(ref,model):
	''' 
	RMSE - Standard Deviation  ratio
	The traditional RMSE method weighted by the standard deviation
	of the observed values (Moriasi et al., 2007)
	'''
	return RMSE(ref,model)/ref.std()