# %%
#The data base used for this is the Human Activity Recgoniyion with Smartphones database, and the MK algorithm is Logistic Regression
#It was build from the recordings of study participants performing activities of daily living (ADL) while carrying a smartphone 
#Objective: classify activities into one of six activities 
    #Walking
    #Walking Upstairs
    #Walking Downstairs
    #Sitting
    #Standing
    #Laying

#For each record in the dataset it is provided:
    #Traxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration
    #Traxial Angular velocity from the gyroscope
    #A 561-feature vector with time and frequency domain variables
    #its activity label


import os
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
# Question 1
    #Import the data and do the following
    #Examine the data if the floating point values need to be scaled 
    #Determine the breakdown of each activity
    #Encode the activity label as an integer
print('\nPreparing to read Human Activity Recognition\n')
human_activity_data = pd.read_csv('Human_Activity_Recognition.csv')
print('\nData set download check\n')
human_activity_data.head()


# %%
#Look at the different data types and recall that when one runs data.dtypes, one gets the data type for each one of the different columns 
print('\nData types in human recognition data:\n', human_activity_data.dtypes)
#In order to see how much of each of the different data types are coming up, one just runs value counts
print('\nData types value counts:\n', human_activity_data.dtypes.value_counts())
#One can observe that there are 561 floats and 1 object (which is what I am trying to predict)
print('\nLooking at the last values:\n',human_activity_data.dtypes.tail())
#We can observe that Activity is the one column with object types 


# %%
print('\nChecking for min values:\n', human_activity_data.iloc[:, :-1].min().value_counts())
#Therefore there's 561 -1 as the minimun value
print('\nChecking for max values:\n', human_activity_data.iloc[:, :-1].max().value_counts())
#One can observe that the data is scaled from a minimum of -1 to a maximum of 1
#So there is some type of scaling here and the way that to prove is that for every single value, see that all of the minimums are -1 and all the maximums are 1
#Next the thing to do is look at the breakdown of each one of the activities 
print('\nBreakdown of activities--they are relatively balanced:\n', human_activity_data.Activity.value_counts())
#The outcome variable has a fairly balanced set
#They wach take up an equal proportion of the overall rows 


# %%
#Different types of error metrics are going to work better for different types of datasets, whether they're balanced or unbalanced 
#In this case the data set is balanced 
#So thinking of what is the best type of error metric to use give that the dataset is rationally balanced

#Since I cannot pass in string into sklearn I have to encode that as an integer

label_encoder = LabelEncoder()
human_activity_data['Activity'] = label_encoder.fit_transform(human_activity_data.Activity)
human_activity_data['Activity'].sample(5)
#So now the activity column has been changed into integers ranging from 0 to 5 for each one of the different categories 

# %%
#Question 2 
    #Calculate the correlations between the independent variables 
    #Create a histogram of the correlation values
    #Identify those that are most correlated (either possitively or negatively)

#Calculate the correlation values
feature_cols = human_activity_data.columns[:-1]
#Will output a pandas dataframe that's just a correlation matrix  
corr_values = human_activity_data[feature_cols].corr()

#Simplify by emptying all the data below the diagonal, since it will not apport any new info
tril_index = np.tril_indices_from(corr_values)#Get all indices from the bottom lower triangle

#Make the unused value NaNs
corr_array = np.array(corr_values)#Change current pandas df to a np array
corr_array[np.tril_indices_from(corr_values)] = np.nan #For the indices that we defined, set them to np.nan

#Recreate correlation pandas dataframe
corr_values = pd.DataFrame(corr_array, columns = corr_values.columns, index = corr_values.index)

#Stack the data and convert to a dat frame
corr_values = (corr_values
               .stack()
               .to_frame()
               .reset_index()
               .rename(columns={'level_0' :'feature1',
                                'level_1': 'feature2',
                                0 : 'correlation'}))
#Get the absolute values for sorting
corr_values['abs_correlation'] = corr_values.correlation.abs()#All we care is about the magnitude of set correlation, not whether it's positive or negative


# %%
#Plot out each of the different correlations 

sns.set_context('talk')
sns.set_style('white')

ax = corr_values.abs_correaltion.hist(bins = 50, figsize = (12, 8))
ax.set(xlabel = 'Absolute Correlation', ylabel = 'Frequency')
ax.show()
# %%
