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
    talphaval = stats.t.isf(alpha, n-2) #double check with ppf
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


'''
Linear Regression
'''


def regresssum(model, alpha=0.05):
    '''
    Linear Regression Summary from statsmodel.
    example:
        X = sm.add_constant(df[['Sales','Market Value']])
        Y = df['Stock Price']
        model = sm.OLS(Y,X).fit()
        regresseq(model)
    :param model: statsmodel object
    :param alpha : confident level. default 0.05
    :return: string of linear regression equation, coef, adjusted coef
    '''
    eqstr = "y = {0}".format(model.params[0])
    for i,xval in enumerate(model.params):
        if i > 0:
            eqstr += " + ({0})x{1}".format(xval, i)
    print(model.summary(alpha=alpha))
    print("{0}Custom Summary{0}".format("="*5))
    print("="*50)
    print("Linear Regression Eq:\t{0}\nCoefficient of determination (R^2):\t{1}\nAdjusted coefficient of determination (adj R^2):\t{2}\n".format(eqstr, model.rsquared, model.rsquared_adj))
    print("="*50)
    return eqstr

