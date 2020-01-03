import math, time, sys, os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.api as smstats


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


def confint_propo(var1, var2, alpha=0.05):
    '''
    Confident interval for proportion
    :param var1: dataframe var 1 (observe)
    :param var2: dataframe var 2 (total_observe)
    :param alpha: significance value
    :return:
    '''
    ci = smstats.proportion_confint(var1, var2, alpha=alpha)
    print(pd.DataFrame({'Lower CI': [ci[0]], 'Upper CI': [ci[1]]}))
    return ci


def confint_mean(var, alpha=0.05, alternative='two-sided'):
    '''
    Confident interval with mean
    :param var: dataframe var 1
    :param alpha: significance level
    :param alternative : h1 != val (two-sided)
                         h1 > val (larger)
                         h1 < val (smaller)
    :return:
    '''
    s = smstats.DescrStatsW(var)
    ci = None
    if s.std:
        ci = s.zconfint_mean(alpha,alternative)
    else:
        ci = s.tconfint_mean(alpha,alternative)
    print("{0}Confidence Interval - Compare Means{0}".format("="*5))
    print("=" * 50)
    print(pd.DataFrame({'Mean': [var.mean], 'Lower CI': [ci[0]], 'Upper CI': [ci[1]]}))
    print("=" * 50)
    return ci


def singletest(var1, mean, alpha=0.05):
    '''
    T-test for the mean of ONE group of scores
    :param var1: dataframe var 1
    :param var2: dataframe var 2
    :param alpha: significance level
    :return:
    '''
    tset, pval = stats.ttest_1samp(var1, mean)
    if pval < alpha:  # alpha value is 0.01 or 1%
        print("We are rejecting null hypothesis", pval)
    else:
        print("We are accepting null hypothesis", pval)
    return pval, alpha


def comparemeans(var1,var2,alpha=0.05):
    '''
    Compare means based on 2 sample datas
    :param var1: dataframe var 1
    :param var2: dataframe var 2
    :param alpha: significance level
    :return:
    '''
    cm = smstats.CompareMeans(smstats.DescrStatsW(var1), smstats.DescrStatsW(var2))
    ci = cm.tconfint_diff(alpha=alpha)
    print("{0}Confidence Interval - Compare Means{0}".format("="*5))
    print("="*50)
    print("LCI:\t{0}\nUCI:\t{1}\n\nconfidence interval:\t{2}".format(ci[0], ci[1], ci))
    print("="*50)
    return ci


def independenttest(var1, var2, alpha=0.05):
    '''
     T-test for the means of TWO INDEPENDENT samples of scores
    :param var1: dataframe var 1
    :param var2: dataframe var 2
    :param alpha: significance level
    :return: p-val, alpha
    '''
    ttest,pval = stats.ttest_ind(var1, var2)
    if pval < alpha:
      print("We are rejecting null hypothesis", pval, ttest)
    else:
      print("We are accepting null hypothesis", pval, ttest)
    return pval,alpha


def relatedtest(var1, var2, alpha=0.05):
    '''
    T-test on TWO RELATED/DEPENDENT samples of scores
    :param var1: dataframe var 1
    :param var2: dataframe var 2
    :param alpha: significance level
    :return: p-val, alpha
    '''
    ttest,pval = stats.ttest_rel(var1, var2)
    if pval < alpha:
      print("We are rejecting null hypothesis", pval, ttest)
    else:
      print("We are accepting null hypothesis", pval, ttest)
    return pval,alpha


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
        p-value < alpha = NOT REJECTING H0 (No correlation)
        p-value > alpha = REJECTING H0 (Correlation exists)
    :param r: sample correlation coefficient . get from df.corr()
    :param n: sample size
    :param alpha: significance level
    :return: reject or not rejecting H0
    '''
    talphaval = stats.t.isf(alpha, n-2) #double check with ppf
    ttestval = coeftest(r, n)
    print("{0}Hypothesis Coefficient Correlation{0}".format("="*5))
    print("="*50)
    print("r:\t{0}\nn:\t{1}\n\nnumber of standard error:\t{2}".format(r, n, ttestval))
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
    example linear:
        X = sm.add_constant(df[['Sales','Market Value']])
        Y = df['Stock Price']
        model = sm.OLS(Y,X).fit()
        regresssum(model)
    example polynomial:
        from sklearn import preprocessing
        X = preprocessing.PolynomialFeatures(2).fit_transform(df['Women'].values.reshape(-1,1))
        Y = df['Men']
        model = sm.OLS(Y,X).fit()
        regresssum(model)
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
    print("Linear Regression Eq:\t{0}\nalpha:\t{1}\nCoefficient of determination (R^2):\t{2}\nAdjusted coefficient of determination (adj R^2):\t{3}\n".format(eqstr, alpha, model.rsquared, model.rsquared_adj))
    print("="*50)
    return eqstr


def hypregresspval(model, var, alpha=0.05):
    '''
    Hypothesis testing for single var regression
        example linear:
        X = sm.add_constant(df[['Sales','Market Value']])
        Y = df['Stock Price']
        model = sm.OLS(Y,X).fit()
        hypregresspval(model)
    example polynomial:
        from sklearn import preprocessing
        X = preprocessing.PolynomialFeatures(2).fit_transform(df['Women'].values.reshape(-1,1))
        Y = df['Men']
        model = sm.OLS(Y,X).fit()
        hypregresspval(model)
    :param model: model
    :param var: variable name
    :param alpha:
    :return:
    '''
    df = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
    pvalue = df.loc[var, 'P>|t|']
    if pvalue > alpha:
        print("p-value({0}) > alpha({1})".format(pvalue, alpha))
        print("Not rejecting H0. Thus model is not significance.")
        return True
    else:
        print("p-value({0}) < alpha({1})".format(pvalue, alpha))
        print("Rejecting H0. Thus model is significance.")
        return False


def hypregresspvalall(model, alpha=0.05):
    '''
    Hypothesis testing for whole regression
    :param model: model
    :param alpha: significance level

    :return:
    '''
    print(model.summary(alpha=alpha))
    if model.f_pvalue > alpha:
        print("p-value({0}) > alpha({1})".format(model.f_pvalue, alpha))
        print("Not rejecting H0. Thus model is not significance.")
    else:
        print("p-value({0}) < alpha({1})".format(model.f_pvalue, alpha))
        print("Rejecting H0. Thus model is significance.")