#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:46:34 2023

@author: asus
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import errors as err
import sklearn.metrics as skmet
#import seaborn as sns
import sklearn.preprocessing as prep
from sklearn import cluster
import sklearn.cluster as cluster

def get_data_frames(filename,countries,indicator):
    '''
    This function returns two dataframes one with countries as column and other 
    one years as column.
    It tanspose the dataframe and converts rows into column and column into 
    rows of specific column and rows.
    It takes three arguments defined as below. 

    Parameters
    ----------
    filename : Text
        Name of the file to read data.
    countries : List
        List of countries to filter the data.
    indicator : Text
        Indicator Code to filter the data.

    Returns
    -------
    df_countries : DATAFRAME
        This dataframe contains countries in rows and years as column.
    df_years : DATAFRAME
        This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by countries
    df = df.loc[df['Country Name'].isin(countries)]
    # To filter data by indicator code.
    df = df.loc[df['Indicator Code'].eq(indicator)]
    
    # Using melt function to convert all the years column into rows as 1 column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Country Code']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code']
                          ,'Country Name').reset_index()
    
    df_countries = df
    df_years = df2
    
    # Cleaning data droping nan values.
    df_countries.dropna()
    df_years.dropna()
    
    return df_countries, df_years

def get_data_frames1(filename,countries,indicator):
    '''
    This function returns two dataframes one with countries as column and other 
    one years as column.
    It tanspose the dataframe and converts rows into column and column into 
    rows of specific column and rows.
    It takes three arguments defined as below. 

    Parameters
    ----------
    filename : Text
        Name of the file to read data.
    countries : List
        List of countries to filter the data.
    indicator : Text
        Indicator Code to filter the data.

    Returns
    -------
    df_countries : DATAFRAME
        This dataframe contains countries in rows and years as column.
    df_years : DATAFRAME
        This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by countries
    # df = df.loc[df['Country Name'].isin(countries)]
    # To filter data by indicator code.
    # df = df.loc[df['Indicator Code'].eq(indicator)]
    
    # Using melt function to convert all the years column into rows as 1 column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Indicator Name']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value',['Years','Country Name','Country Code']
                          ,'Indicator Code').reset_index()
    
    df_countries = df
    df_indticators = df2
    
    # Cleaning data droping nan values.
    df_countries.dropna()
    df_indticators.dropna()
    
    return df_countries, df_indticators


def poly(x, a, b, c, d):
    '''
    Cubic polynominal for the fitting
    '''
    y = a*x*3 + b*x*2 + c*x + d
    return y

def exp_growth(t, scale, growth):
    ''' 
    Computes exponential function with scale and growth as free parameters
    '''
    f = scale * np.exp(growth * (t-1960))
    return f

def logistics(t, scale, growth, t0):
    ''' 
    Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    '''
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def norm(array):
    '''
    Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe
    '''
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled

def norm_df(df, first=0, last=None):
    '''
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    '''
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df


def map_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns???
    ??????in the dataframe.
    Input:
    df: pandas DataFrame
    size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='magma')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)


#==============================================================================
# Data fitting for China Population with prediction
#==============================================================================

countries = ['Germany','Australia','United States','China','United Kingdom']
# calling functions to get dataframes and use for plotting graphs.
df_c, df_y = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv',countries,
                             'SP.POP.TOTL')

df_y['Years'] = df_y['Years'].astype(int)

popt, covar = curve_fit(exp_growth, df_y['Years'], df_y['China'])
print("Fit parameter", popt)
# use *popt to pass on the fit parameters
df_y['china_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y["China"], label='data')
plt.plot(df_y['Years'], df_y['china_exp'], label='fit')
plt.legend()
plt.title("First fit attempt")
plt.xlabel("Year")
plt.ylabel("China Population")
plt.show()

# find a feasible start value the pedestrian way
# the scale factor is way too small. The exponential factor too large.
# Try scaling with the 1950 population and a smaller exponential factor
# decrease or increase exponential factor until rough agreement is reached
# growth of 0.07 gives a reasonable start value
popt = [7e8, 0.01]
df_y['china_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['China'], label='data')
plt.plot(df_y['Years'], df_y['china_exp'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("China Population")
plt.title("Improved start value")
plt.show()

# fit exponential growth
popt, covar = curve_fit(exp_growth, df_y['Years'],df_y['China'], p0=[7e8, 0.02])
# much better
print("Fit parameter", popt)
df_y['china_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['China'], label='data')
plt.plot(df_y['Years'], df_y['china_exp'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("China Population")
plt.title("Final fit exponential growth")
plt.show()


# estimated turning year: 1990
# population in 1990: about 1135185000
# kept growth value from before
# increase scale factor and growth rate until rough fit
popt = [1135185000, 0.02, 1990]
df_y['china_log'] = logistics(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['China'], label='data')
plt.plot(df_y['Years'], df_y['china_log'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("China Population")
plt.title("Improved start value")
plt.show()

popt, covar = curve_fit(logistics,  df_y['Years'],df_y['China'],
p0=(6e9, 0.05, 1990.0))
print("Fit parameter", popt)
df_y['china_log'] = logistics(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['China'], label='data')
plt.plot(df_y['Years'], df_y['china_log'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("China Population")
plt.title("Logistic Function")


# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)

low, up = err.err_ranges(df_y['Years'], logistics, popt, sigma)
plt.figure()
plt.title("logistics function")
plt.plot(df_y['Years'], df_y['China'], label='data')
plt.plot(df_y['Years'], df_y['china_log'], label='fit')
plt.fill_between(df_y['Years'], low, up, alpha=0.7)
plt.legend()
plt.xlabel("Year")
plt.ylabel("China Population")
plt.show()

print("Forcasted population")
low, up = err.err_ranges(2030, logistics, popt, sigma)
print("2030 between ", low, "and", up)
low, up = err.err_ranges(2040, logistics, popt, sigma)
print("2040 between ", low, "and", up)
low, up = err.err_ranges(2050, logistics, popt, sigma)
print("2050 between ", low, "and", up)

print("Forcasted population")
low, up = err.err_ranges(2030, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)



#==============================================================================
# Data fitting with ouliners for Total Population
#==============================================================================
# List of countries 
countries = ['Germany','Australia','United States','China','United Kingdom']
# calling functions to get dataframes and use for plotting graphs.
df_c, df_y = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv',countries,
                             'SP.POP.TOTL')


df_c.dropna()
df_y.dropna()


df_y['Years'] = df_y['Years'].astype(int)
x = df_y['Years'].values
y = df_y['China'].values 
z = df_y['United States'].values
w = df_y['United Kingdom'].values 

param, covar = curve_fit(poly, x, y)
# produce columns with fit values
df_y['fit'] = poly(df_y['Years'], *param)
# calculate the z-score
df_y['diff'] = df_y['China'] - df_y['fit']
sigma = df_y['diff'].std()
print("Number of points:", len(df_y['Years']), "std. dev. =", sigma)
# calculate z-score and extract outliers
df_y["zscore"] = np.abs(df_y["diff"] / sigma)
df_y = df_y[df_y["zscore"] < 3.0].copy()
print("Number of points:", len(df_y['Years']))

param1, covar1 = curve_fit(poly, x, z)
param2, covar2 = curve_fit(poly, x, w)

plt.figure()
plt.title("Total Population (Data Fitting)")
plt.scatter(x, y, label='China')
plt.scatter(x, z, label='Australia')
plt.scatter(x, w, label='United Kingdom')
plt.xlabel('Years')
plt.ylabel('Total Population')
x = np.arange(1960,2021,10)
plt.plot(x, poly(x, *param), 'k')
plt.plot(x, poly(x, *param1), 'k')
plt.plot(x, poly(x, *param2), 'k')
plt.xlim(1960,2021)
plt.legend()
plt.show()



