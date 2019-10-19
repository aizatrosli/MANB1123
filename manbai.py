import math, time, sys, os
import pandas as pd
import numpy as np
from scipy import stats

def errormean(zvalue,n,m):
	return (zvalue*m)/math.sqrt(n)

def errorpropotion(zvalue,n,p):
	return (zvalue*math.sqrt((p*(1-p))/n))

def intervalmarginerror(mean,error):
	return [mean+error,mean-error]