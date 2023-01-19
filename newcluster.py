#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:33:37 2023

@author: asus
"""
# Importing libraries
import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.cluster import KMeans
import seaborn as sns
import itertools as iter

def read(indicator, country_code):
    '''
     This function is used to get the dataframe. 
     Indicator and country_code are passed as the parameters. 
    '''
    df = wb.data.DataFrame(indicator, country_code, mrv=30)
    return df

#==============================================================================
def norm_df(df):
    '''
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    '''
    y = df.iloc[:, 2:]
    df.iloc[:, 2:] = (y-y.min()) / (y.max() - y.min())
    return df

#==============================================================================
def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    """
# initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    uplow = []
# list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        pmix = list(iter.product(*uplow))
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    return lower, upper

#==============================================================================
def fct(x, a, b, c):
    '''
     polynominal function for the fitting
    '''
    return a*x**2+b*x+c
#==============================================================================
# Indicators selected
#--------------Indicator 1 includes CO2 emissions (metric tons per capita) and
# Electric power consumption (kWh per capita)---------------------------------#
#---------------Indicator 2 includes Methane emissions (kt of CO2 equivalent) 
# and Access to electricity (% of population)---------------------------------#
indicator1 = ["EN.ATM.CO2E.PC", "EG.USE.ELEC.KH.PC"]
indicator2 = ["EN.ATM.METH.KT.CE", "EG.ELC.ACCS.ZS"]

# Countries selected [Australia, China, Germany and United States]
country_code = ['AUS', 'CHN', 'DEU', 'USA']

# The CSV file is stored in a variable "Path".
path = "World Indicator Repository.csv"
# Indicator1 and country code read into "dat".
dat = read(indicator1, country_code)
dat.columns = [i.replace('YR', '') for i in dat.columns]
dat = dat.stack().unstack(level=1)
# Setting the column name
dat.index.names = ['Country', 'Year']
dat.columns
dat1 = read(indicator2, country_code)
dat1.columns = [i.replace('YR', '') for i in dat1.columns]
dat1 = dat1.stack().unstack(level=1)
dat1.index.names = ['Country', 'Year']
dat1.columns
# Reseting the index.
dt1 = dat.reset_index()
dt2 = dat1.reset_index()
dt = pd.merge(dt1, dt2)
# Cleaning data droping nan values.
dt.dropna()
dt.drop(['EG.USE.ELEC.KH.PC'], axis=1, inplace=True)
dt.drop(['EG.ELC.ACCS.ZS'], axis=1, inplace=True)
dt
dt["Year"] = pd.to_numeric(dt["Year"])
dt_norm = norm_df(dt)
print(dt_norm)
# Droping the column countries
df_fit = dt_norm.drop('Country', axis=1)
print(df_fit)

# Plot for two clusters
k = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(df_fit)
# Seaborn is used to scatterplot
sns.set(style='whitegrid')
sns.scatterplot(data=dt_norm, x="Country", y="EN.ATM.CO2E.PC",hue=k.labels_ )
# Legend displayed
plt.title("2 Clusters of countries for Co2 Emission")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

dt1 = dt[(dt['Country'] == 'AUS')]
dt1
val = dt1.values
x, y = val[:, 1], val[:, 2]

# Curve fit for the country Australia.
# Indicator used is Co2 emission
prmet, cov = opt.curve_fit(fct, x, y)
dt1["pop_log"] = fct(x, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x, dt1["pop_log"], label="Fit")
plt.plot(x, y, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in Australia")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x, fct, prmet, sigma)
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
dt2 = dt[(dt['Country'] == 'USA')]
dt2
val2 = dt2.values
x2, y2 = val2[:, 1], val2[:, 2]

# Curve fit for the country USA.
# Indicator is Co2 emmision
prmet, cov = opt.curve_fit(fct, x2, y2)
dt2["pop_log"] = fct(x2, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x2, dt2["pop_log"], label="Fit")
plt.plot(x2, y2, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.title("CO2 emission rate in USA")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x2, fct, prmet, sigma)
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
dt3 = dt[(dt['Country'] == 'CHN')]
dt3
val3 = dt3.values
x3, y3 = val3[:, 1], val3[:, 2]

# Curve fit for the coutry China
# Indicator used is Co2 emmision
prmet, cov = opt.curve_fit(fct, x3, y3)
dt3["pop_log"] = fct(x3, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x3, dt3["pop_log"], label="Fit")
plt.plot(x3, y3, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.title("CO2 emission rate in China")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x3, fct, prmet, sigma)
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
