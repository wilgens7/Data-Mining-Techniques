import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.manifold import MDS


pd.set_option('display.max_columns', 9)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

#======================LOADING DATA INTO DATAFRAME====================
data = pd.read_csv('monroe-county-crash-data2003-to-2015.csv',encoding='latin1')

#print(data.head())
#======================PREPROCESSING==================================

#1 drop Master Record Number colum
data = data.drop('Master Record Number',axis = 1)

print(data.columns)

#2 changing Weekend column entries to 0 and 1 

data['Weekend?'][data['Weekend?']=='Weekday'] = 0
data['Weekend?'][data['Weekend?']=='Weekend'] = 1

#print(data['Weekend?'])

#3 dropping records with blank hours

#print(len(data.Hour[data.Hour.isna()]))
#print(len(data))

data = data.dropna(subset=['Hour'])

#print(len(data.Hour[data.Hour.isna()]))

#print(len(data))

#4 droping invalid latitude and longitude

print(len(data[(data.Latitude == 0) 
	|(data.Latitude == 1) 
	| (data.Longitude == 0)
	|(data.Longitude == 1)
	|(data.Longitude.isna()
	|(data.Latitude.isna()))]))

#print(len(data))

data = data.drop(data[(data.Latitude == 0) 
	|(data.Latitude == 1) 
	| (data.Longitude == 0)
	|(data.Longitude == 1)
	|(data.Longitude.isna()
	|(data.Latitude.isna()))].index)

#print(len(data))

#5 droping Primary Factor Row
data = data.drop('Primary Factor',axis = 1)

#6 droping Primary Report_Location
data = data.drop('Reported_Location',axis = 1)

print(data.columns)


#7 encoding Collision Type and Injury Type

#a Collision Types - one-hot encoding 

collision_types = pd.get_dummies(data['Collision Type'])

#print(collision_types)

data = pd.concat([data,collision_types],axis = 'columns')

#b droping original column and one dependent column
data = data.drop(['Collision Type','Bus'],axis = 1)

#print(data.head())

#print(len(data))

#c Injury Type
#print(data['Injury Type'].unique())

#data['Injury Type'][data['Injury Type']=='No injury/unknown'] = len(data['Injury Type'][data['Injury Type']=='No injury/unknown'])
#data['Injury Type'][data['Injury Type']=='Non-incapacitating'] = len(data['Injury Type'][data['Injury Type']=='Non-incapacitating'])
#data['Injury Type'][data['Injury Type']=='Incapacitating'] = len(data['Injury Type'][data['Injury Type']=='Incapacitating'])
#data['Injury Type'][data['Injury Type']=='Fatal'] = len(data['Injury Type'][data['Injury Type']=='Fatal'])

injury_types = pd.get_dummies(data['Injury Type'],drop_first = True)

#injury_types = injury_types.drop('Fatal',axis = 'columns')

data = pd.concat([data,injury_types],axis = 'columns')

data = data.drop('Injury Type', axis = 'columns')

print(data.head())

#d making injury types numeric
#data['Injury Type']=pd.to_numeric(data['Injury Type'])

#===============================Descriptive Statistics======================

#1 general descriptive analysis 
#print(data.describe()) #for report

#2 mode of each category
#for i in data.columns:
	#print(i,":",data[i].mode()) #for report

# Covariance matrix 
print(data.cov())# for report 

#4 Correlation matrix
print(data.corr())# for report




#==============================Figures=========================================
# Heatmap

cor = data.corr()

plt.figure(figsize = (17,13))
plt.rcParams.update({'font.size': 6})

ax = sns.heatmap(
    cor, 
    annot = True,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

# Histograms
sns.distplot(data['Hour'])
plt.show()
plt.hist(data['Hour'])
plt.show()

#===============================Categorical Prediction====================

#cross val
#cv=ShuffleSplit(n_splits=20,test_size=0.3,train_size=0.7,random_state=None)
#score = cross_val_score(model,X,y,cv=cv,scoring='r2')

#===============================Clustering Prediction=====================







