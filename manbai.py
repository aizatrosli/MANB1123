import math, time, sys, os
import pandas as pd
import numpy as np
from scipy import stats

'''
Estimation
'''
def pointestimate(x,n):
	'''
	point estimate
		x/n	OR	X/N
	:param x: number of items interest
	:param n: size of population
	:return: population proportion OR sample proportion
	'''
	pe = x/n
	print("{0}Margin of Error (mean population){0}".format("="*5))
	print("="*50)
	print("x:\t{0}\nn:\t{1}\npe:\t{2}".format(x, n, pe))
	print("="*50)
	return pe

def errormeanpop(cl, std, n, tail=1):
	'''
	This margin of error mean population
		z∗(σ/√n)
	*reminder if 95% confidence level:
		- two-tail : q=0.975
		- one-tail : q=0.95
	:param cl: confidence level. eg 95%
	:param std: standard deviation. use std()
	:param n: sample size
	:param kwargs: tail = specify tail number. default: 1
	:return: margin of error mean population
	'''
	q = 1-((1-(cl/100))/2) if tail != 1 else cl/100
	zcritical = stats.norm.ppf(q=q)
	merror = (zcritical*std)/math.sqrt(n)
	print("{0}Margin of Error (mean population){0}".format("="*5))
	print("="*50)
	print("zcritical:\t{0}\nstd:\t{1}\nn:\t{2}\n\nmargin of error:\t{3}".format(zcritical,std,n,merror))
	print("="*50)
	return merror


def errorpopproportion(cl, p, n, tail=1):
	'''
	Margin of error mean population
			_______
		z∗ /p(1--p)
		 / --------
	   √      n

	*reminder if 95% confidence level:
		- two-tail : q=0.975
		- one-tail : q=0.95
	:param cl: confidence level. eg 95%
	:param p: population propotion
	:param n: sample
	:param tail: specify tail number. default: 1
	:return: margin of error population proportion
	'''
	print("{0}Margin of Error (population proportion){0}".format("="*5))
	print("="*50)
	q = 1-((1-(cl/100))/2) if tail != 1 else cl/100
	zcritical = stats.norm.ppf(q=q)
	merror = zcritical * math.sqrt((p * (1 - p)) / n)
	print("zcritical:\t{0}\np:\t{1}\nn:\t{2}\n\nmargin of error:\t{3}".format(zcritical, p, n, merror))
	print("="*50)
	return merror


def errorsinglepopmeans(cl, std, n, tail=1):
	'''
	Margin of error mean population
		t∗(σ/√n)
	*reminder if 95% confidence level:
		- two-tail : q=0.975
		- one-tail : q=0.95
	:param cl: confidence level. eg 95%
	:param std: sample std,s = df.std(ddof=1)
	:param n: sample
	:param tail: specify tail number. default: 1
	:return:
	'''
	print("{0}Margin of Error (single population means T-test){0}".format("="*5))
	print("="*50)
	q = 1-((1-(cl/100))/2) if tail != 1 else cl/100
	tcritical = stats.t.ppf(q=q, df=n-1)
	merror = (tcritical * std) / math.sqrt(n)
	print("tcritical:\t{0}\nstd:\t{1}\nn:\t{2}\n\nmargin of error:\t{3}".format(tcritical,std,n,merror))
	print("="*50)
	return merror

def cinterval(pe, merror):
	'''
	confidence interval = point estimate * margin of error
		1) population proportion
			cinterval(pe, errorpoppropotion())
		2) single population means
			cinterval(pe, errorsinglepopmeans())
	:param pe: sample mean / sample proportion / population proportion / population mean
	:param merror: margin of error mean population or population proportion.
					please run errormeanpop or errorpopproption
	:return: confindence interval
	'''
	ci = [pe-merror, pe+merror]
	print("{0}Confidence Interval{0}".format("="*5))
	print("="*50)
	print("point estimation:\t{0}\nmargin of error:\t{1}\n\nconfidence interval:\t[{0}+{1} ,{0}-{1}]\n\t\t\t[{2} , {3}]".format(pe, merror, ci[0],ci[1]))
	print("="*50)
	return ci


'''
Hypothesis Testing
'''

'''
Correlation
'''


def coeftest(r, n):
	'''
	T-test for correlation
	:param r: sample correlation coefficient . get from df.corr()
	:param n: sample size
	:return: number of standard errors is from 0
	'''
	t = r / math.sqrt((1 - math.pow(r, 2)) / (n - 2))
	print("{0}Correlation Testing{0}".format("="*5))
	print("="*50)
	print("r:\t{0}\nn:\t{1}\n\nnumber of standard error:\t{2}".format(r, n, t))
	print("="*50)
	return t


def hypcoeftest(r, n, alpha=0.05):
	'''
	hypothesis coefficient t-test for correlation

	:param r: sample correlation coefficient . get from df.corr()
	:param n: sample size
	:param alpha: significance level
	:return: reject or not rejecting H0
	'''
	talphaval = stats.t.isf(alpha, n-2)
	ttestval = coeftest(r, n)
	print("{0}Hypothesis Coefficient Correlation{0}".format("="*5))
	print("="*50)
	print("r:\t{0}\nn:\t{1}\n\nnumber of standard error:\t{2}".format(r, n, t))
	print("="*50)
	if ttestval < talphaval:
		print("p-value({0}) < alpha({1})".format(ttestval, talphaval))
		print("Not rejecting H0. There is no positive correlation exists.")
		return False
	else:
		print("p-value({0}) > alpha({1})".format(ttestval, talphaval))
		print("Rejecting H0. Positive correlation exists between 2 variables.")
		return True


class bstat(object):

	def __init__(self):
		self.level = 0
		self.dict = {}


	def whybs(self):
		'''
		why business statistics:
		- know about the business
		- decision making
		- competitiors
		:return:
		'''
		return self.dict['why']

	def paramvsstat(self):
		'''
		parameter vs statistic
		- population | sample
		- propotion | mean
		- all students | sample of 100 students
		:return: dictionary parameter vs statistic
		'''
		return self.dict['paramvsstat']

	def varvsdata(self):
		'''
		variable vs data
		- characteristic of an item | observed value of a variable
		- a piece of data contain 1 or more val | data have variable as col and obs as row
		:return: dictionary variable vs data
		'''
		return self.dict['varvsdata']

	def var(self):
		'''
		variable
		- Quantitative (numerical) var
			- Discrete var : finite / countable | val result from counting
			- Continuous var : infinite / not countable | val is measured
		- Qualitative (categorical) var
		:return:
		'''
		return self.dict['var']

	def datalvl(self):
		'''
		data measurement/data hierarchy/data level
		- Nominal data
		- Ordinal data / Rank data
		- Interval data
		- Ratio data
		:return:
		'''
		return self.dict['datalevel']

	def datalvlissue(self):
		'''
		data-level issues
		- Mean [Ratio,Interval]
			- Numerical center of the data
			- Sum of deviation from the mean is zero
			- Sensitive to outliers / not skewed
		- Median [Ratio,Interval,Ordinal]
			- Not sensitive to outliers / skewed
			- Computed only from the center val
			- Does not use info from all data
		- Mode [Ratio,Interval,Ordinal,Nominal]
			- May not reflect / exist
			- Might have multiple modes
		:return:
		'''
		return self.dict['datalvlissue']

	def sourcedata(self):
		'''
		sources of data
		- Published Sources (print or electronic form)
			- primary data source : published by individual or group.
			- secondary data source : compiled from primary sources.
		- Surveys (questionnaire or responses)
			- informal survey : open to everyone. (many variable)
			- targeted survey : directed to specific group. (less variable)
		- Experiment (examine effect variable and keep other things equal)
			- treatment group : receive the treatment.
			- control group : not receive the treatment.

		:return:
		'''
		return self.dict['sourcedata']

	def popvssample(self):
		'''
		population vs sample
		- population : entire group.
		- sample : subset of the population.
		- individual : member of the population.
		- census : list of all individual in population with characteristics.
		:return:
		'''
		return self.dict['popvssample']

	def sampletype(self):
		'''
		sampling type
		- non statistical sampling (subjective judgement)
			- convenience | judgmental | volunteer | snowball
		- statistical sampling (population to have a known or calculable)
			- simple random | stratified random | systematic | cluster
		:return:
		'''
		return self.dict['sampletype']

	def missdata(self):
		'''
		type of missing data
		- missing completely at random (MCAR)
			- does not depend on the values.
			- ignore missing data without bias.
		- missing at random (MAR)
			- does not depend on the unobserved values.
			- but does depend on the observed.
		- not missing at random (NMAR)
			- depends on the unobserved values
		:return:
		'''
		return self.dict['missingdata']

	def handlemissdata(self):
		'''
		handle the missing data
		- deletion methods
			- list-wise deletion : remove whole row data while produces balanced data but less data
			- pairwise deletion : ??
		- imputation methods
			- simple imputation
				- random sample
				- mean, median, mode
			- multiple imputation
		:return:
		'''
		return self.dict['handlemissdata']