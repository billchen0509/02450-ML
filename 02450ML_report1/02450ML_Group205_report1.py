# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:41:08 2023

@author: Group 205
"""

#%%
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
import matplotlib.pyplot as plt
from scipy.linalg import svd

#%%
# load data
path = 'forest_fire.csv'

df = pd.read_csv(path, header = 1)

df = df.drop(index = [122,123,124])

df = df.drop(columns = ['day','month','year'])

df.columns.values[0] = 'Temp'

raw_data = df.values

#%% 
cols = range(0,10)
X = raw_data[:, cols]

# convert X string array to float array
X = np.asfarray(X,dtype = float)

# extract the attribute names 
attributeNames = np.asarray(df.columns[cols])

# extract the last column
classLabels = raw_data[:,-1]

df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: x.strip())


#determine the class labels
classNames = np.flip(np.unique(classLabels))

classDict = dict(zip(classNames,range(len(classNames))))

# Fire or not
y = np.array([classDict[cl] for cl in classLabels])


N,M = X.shape

C = len(classNames)
#%% Plot1: use boxplot to find the outliers

# Create a box plot from the data matrix
plt.boxplot(X)

# Set plot title and axis labels
plt.title('Box Plot of dataset')
plt.xlabel('Variables')
plt.ylabel('Values')
plt.savefig('Plot1: boxplot.png', dpi=300)
# Display the plot
plt.show()

#%% Plot2: Temperature againt relative humidity

c_tem = 0 #column of temp 
c_RH = 1 #column of RH
f = figure()
title('Temperature againt relative humidity')

means = []
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot([float(i) for i in X[class_mask,c_tem]], [float(i) for i in X[class_mask,c_RH]], 'o',alpha=.3)

legend(classNames)
xlabel(attributeNames[c_tem])
ylabel(attributeNames[c_RH])
ystickstep = 20
plt.savefig('Plot2: Temperature againt relative humidity.png', dpi=300)
# Output result to screen
show()

#%% Plot3: Variance explained by principal components

# Data standardization
Y = X - np.ones((N,1))*X.mean(axis=0)
Y = Y / np.std(Y,0)

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.82

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.savefig('Plot3: Variance explained by principal components.png', dpi=300)
plt.show()

#%% Plot4: Forest fire: 2-dimension PCA
#Plot principal component 1 and 2 against each other in a scatterplot
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T  

# Project the centered data onto principal component space
Z = Y @ V

# Plot PCA of the data
fig, ax = plt.subplots()

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,0], Z[class_mask,1], 'o', alpha=.5)
    
mean_vector = np.mean(Y, axis=1)
pc2 = V[:, :2]

# ax.quiver(mean_vector[0], mean_vector[1],
#           pc2[0, 0], pc2[1, 0], color='r', label='PC1', scale=5)
# ax.quiver(mean_vector[0], mean_vector[1],
#           pc2[0, 1], pc2[1, 1], color='r', label='PC2', scale=5)

v1 = ax.quiver(0, 0, pc2[0, 0], pc2[1, 0], color='r', label='v1', scale=2)
v2 = ax.quiver(0, 0, pc2[0, 1], pc2[1, 1], color='g', label='v2', scale=2)
ax.annotate('v1', (2, -2), xytext=(6, 10), textcoords='offset points')
ax.annotate('v2', (-2, 2.5), xytext=(2, 2), textcoords='offset points')


ax.set_title('Forest fire: 2-dimension PCA')
ax.set_xlabel('Xv1')
ax.set_ylabel('Xv2')

legend(classNames)
plt.savefig('Plot4: Forest fire_2-dimension PCA.png', dpi=300)
# Output result to screen
show()

#%% Plot5: Forest fire: 3-dimension PCA

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection = '3d')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    ax.scatter(Z[class_mask,0], Z[class_mask,1],
                   Z[class_mask,2], 'o', alpha=.5)

legend(classNames)

pc3 = V[:, :3]

# ax.quiver(mean_vector[0], mean_vector[1], mean_vector[2],
#           pc3[0, 0], pc3[1, 0], pc3[2, 0], color='r', label='v1',
#           length=5, normalize=True)
# ax.quiver(mean_vector[0], mean_vector[1], mean_vector[2],
#           pc3[0, 1], pc3[1, 1], pc3[2, 1], color='r', label='v2', 
#           length=5, normalize=True)
# ax.quiver(mean_vector[0], mean_vector[1], mean_vector[2],
#           pc3[0, 2], pc3[1, 2], pc3[2, 2], color='r', label='v3',
#           length=5, normalize=True)

v1_ = ax.quiver(0, 0, 0, pc3[0, 0], pc3[1, 0], pc3[2, 0], color='r', label='v1',
          length=10)
v2_ = ax.quiver(0, 0, 0, pc3[0, 1], pc3[1, 1], pc3[2, 1], color='g', label='v2', 
          length=8)
v3_ = ax.quiver(0, 0, 0, pc3[0, 2], pc3[1, 2], pc3[2, 2], color='b', label='v3',
          length=8)

ax.view_init(elev=30, azim=45)
ax.set_xlabel('Xv1')
ax.set_ylabel('Xv2')
ax.set_zlabel('Xv3')

ax.set_xlim([-5, 5]) 
ax.set_ylim([-5, 5])  
ax.set_zlim([-5, 5])

title('Forest fire: 3-dimension PCA')
plt.savefig('Plot5: Forest fire_3-dimension PCA.png', dpi=300)
# Output result to screen
show()

#%% Plot6: Forest Fire: PCA Component Coefficients
# principal directions (V) obtained using the PCA. 

# We saw that the first 3 components explaiend more than 82
# percent of the variance. 
# now look at their coefficients:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Forest Fire: PCA Component Coefficients')
plt.savefig('Plot6: Forest Fire_PCA Component Coefficients.png', dpi=300)
plt.show()





