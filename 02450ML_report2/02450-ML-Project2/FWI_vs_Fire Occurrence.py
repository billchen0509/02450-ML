# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 23:56:55 2023

@author: L Bz
"""

#%% import packages
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#%% load data
path = 'forest_fire.csv'
df = pd.read_csv(path, header = 1)
df = df.drop(index = [122,123,124])
df = df.drop(columns = ['day','month','year'])
df.columns.values[0] = 'Temp'
raw_data = df.values

#%% 
cols = range(9,10)
FWI = raw_data[:, cols]

# convert X string array to float array
FWI = np.asfarray(FWI,dtype = float)

# extract the last column
classLabels = raw_data[:,-1]
df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: x.strip())

#determine the class labels
classNames = np.flip(np.unique(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])

#%% scatter plot
plt.scatter(FWI, y, c=y, cmap='bwr')
plt.xlabel('FWI')
plt.ylabel('Fire Occurrence')
plt.title('FWI vs. Fire Occurrence')
plt.yticks([0, 1])
plt.gca().set_yticklabels(['not fire', 'fire'])
plt.show()






