#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 04:30:12 2023

@author: asus
"""
"""
Imported required libraries
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt



def get_data_frames(filename):
    
    """
    The function get_data_frames takes filename as the 
    argument read the data file
    into dataframes and returns df(countries), df2(years).
    
    """ 
    df_c = pd.read_csv(filename, skiprows=(4), index_col=False)
    print(df_c.info())
    
       
#Statitical function returns the dercription of data in the dataframe.
    print(df_c.describe())

#Removing the column containg "unnamed" as part of the data cleaning.
    df_c = df_c.loc[:, ~df_c.columns.str.contains('^Unnamed')]
   
#Selecting the countries.
   # df_c = df_c.loc[df_c['Country Name'].isin(countries)]

#Transposing the data
    df_y = df_c.melt(id_vars=['Country Name','Country Code',
                           'Indicator Name','Indicator Code'], 
                  var_name='Years')

#Deleted country code.    
    del df_y['Country Code']
    
# Transposing the data
    df_y = df_y.pivot_table('value',
                          ['Years','Indicator Name','Indicator Code'],
                          'Country Name').reset_index()
    
    print(df_y.info())
    
#Statatical function return the description of data in the dataframe
    print(df_y.describe())
    


#Return countries and years.
    return df_c, df_y

#Removes all the rows cotaining the null values.
    
    df_c.dropna()
    df_y.dropna()

#The required countries for ploting the graphs are selected.
#countries = ['Japan','Germany','Canada','United Kingdom']
def poly(x, a, b, c, d):
    """Cubic polynominal for the fitting"""
    
    y = a*x**3 + b*x**2 + c*x + d
    
    return y


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    
    f = scale * np.exp(growth * (t-1950)) 
    
    return f
        

#Read the file into dataframes.
df_c, df_y = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv')

#The function the indicators
df_y = df_y.loc[df_y['Indicator Code'].eq('SP.URB.TOTL')]

df_y['Years'] = df_y['Years'].astype(int)
x = df_y['Years'].values
y = df_y['United Kingdom'].values 

popt, covar = opt.curve_fit(exp_growth, x,y)

param, covar = opt.curve_fit(poly, x, y)
# produce columns with fit values
df_y["fit"] = poly(df_y['Years'], *param)

df_y["fit"] = exp_growth(df_y['Years'], *popt)
# calculate the z-score
df_y["diff"] = df_y['Japan'] - df_y["fit"]
sigma = df_y["diff"].std()
print("Number of points:", len(df_y['Years']), "std. dev. =", sigma)
# calculate z-score and extract outliers
df_y["zscore"] = np.abs(df_y["diff"] / sigma)
df_y = df_y[df_y["zscore"] < 3.0].copy()
print("Number of points:", len(df_y['Years']))

plt.figure()
plt.title("*Urban Population*")
plt.scatter(x, y, label='United Kingdom')
plt.xlabel('Years')
plt.ylabel('Urban Population')
x = np.arange(1960,2021,10)
plt.plot(x, poly(x, *param), 'k')
plt.xlim(1960,2021)
plt.legend()
plt.show()


