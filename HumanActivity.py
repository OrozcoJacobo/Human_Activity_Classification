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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
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

ax = corr_values.abs_correlation.hist(bins = 50, figsize = (12, 8))
ax.set(xlabel = 'Absolute Correlation', ylabel = 'Frequency')
#See that most of the time there's essentialy zero correlation and we see much less close to 1, fairly uniform across once it dips down from those lower values 


# %%
#The most correlated values
#Sort the values by correlation with ascending equals false
corr_values.sort_values('correlation', ascending = False).query('abs_correlation > 0.8')#.query is a way to filter dowwn our database
#Observe that the resulting data has 22815 that are in line with our needs, keep in mind that is out of 157,000 different values
#Also important to note about the most correlated values presented previously is that we may want to do some type of feature engineering or feature selection-
#for those values that are so highly correlated 


# %%
#Question 3
    #Split the data into train and test data sets. This can be done using any method, but consider Scikit-learn's StratifiedShuffleSplit to maintain-
    #the same ratio fo predictior classes
    #Regardless of methods used to split the data, compare the ratio of classes in both the train and test splits

#Get the split indexes
strat_shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
#Next is necesarry since the strat_shuffle_split is going to return a generator object
train_idx, test_idx = next(strat_shuffle_split.split(human_activity_data[feature_cols], human_activity_data.Activity))

#Create the dataframe
X_train = human_activity_data.loc[train_idx, feature_cols]
y_train = human_activity_data.loc[train_idx, 'Activity']

X_test = human_activity_data.loc[test_idx, feature_cols]
y_test = human_activity_data.loc[test_idx, 'Activity']
#Calling notmalize=True withing the value_counts=>They give the proportion, rather than the actual counts, and that will allow for the comparison between
#both y_train and t_test (they are similar for each one of the different activies which are labeled by integers that I want to predict)
print('\ny_train value counts:\n', y_train.value_counts(normalize = True))
print('\ny_test value counts:\n', y_test.value_counts(normalize = True))


# %%
#Question 4
    #Fit a logistic regression model, without any regularization using all of the features. 
    #Using cross_validation to determine the hyperparameters, fit models using L1, and L2 regularization. Store each of these models as well.

#Standard logistic regression
logistic_regression = LogisticRegression(solver='liblinear').fit(X_train, y_train)


# %%
#L1 regularized logistic regression
logistic_regression_l1 = LogisticRegressionCV(Cs = 10, cv=4, penalty='l1', solver='liblinear').fit(X_train, y_train)


# %%
#L2 regularized logistic regression
logistic_regression_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2', solver='liblinear').fit(X_train, y_train)


# %%
#Question 5
#Compare the magnitudes of the coefficients for each of the models. If one-vs-rest fitting was used, each set of coefficients can be plotted separately

#Combine all the coefficients into a dataset
coefficients = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [logistic_regression, logistic_regression_l1, logistic_regression_l2]

for lab, mod in zip(coeff_labels, coeff_models):
    coeffs = mod.coef_
    coeff_labels = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]],
                                 codes=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_labels))

coefficients = pd.concat(coefficients, axis=1)
coefficients.sample(10)


# %%
fig, axList = plt.subplots(nrows = 3, ncols = 2)
axList = axList.flatten()
fig.set_size_inches(10,10)

for loc, ax in enumerate(axList):
    data = coefficients.xs(loc, level=1, axis=1)
    data.plot(marker = 'o', ls ='', ms = 2.0, ax=ax, legend = False)

    if ax is axList[0]:
        ax.legend(loc = 4)

    ax.set(title='Coefficient Set '+ str(loc))

plt.tight_layout()


# %%
#Question 6
    #Predict and store the class for each model
    #Store the probability for the predicted class for each model

#Predict the class and the probability for each

y_pred = list()
y_prob = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [logistic_regression, logistic_regression_l1, logistic_regression_l2]

for lab, mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(X_test), name=lab))
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]],
                                codes=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    
    y_prob.append(pd.DataFrame(mod.predict_proba(X_test), columns=coeff_label))

y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)

print('\ny prediction:\n', y_pred.head())
print('\ny probability:\n', y_prob.head())
print('\nIn order to look at results that arent the same across the board:\n', y_pred[y_pred.lr != y_pred.l1])


# %%
#So now I have probasbilites and predicted labels, so it's time to start coming up with the scores that we'd want in order to actually see how well we performed 
#Question 7
    #For each model, calculate the following error metrics:
        #accuracy
        #precision
        #recall
        #fscore
        #confusion matrix
    #Decide how to combine the multi-class metrics into a single value for each model.

metrics = list()
cm = dict()

for lab in coeff_labels:
    #Precision, recall, f-score from the multi-class support function
    precision, recall, fscore, _ = score(y_test, y_pred[lab], average='weighted')

    #The usual way to calculate accuracy
    accuracy= accuracy_score(y_test, y_pred[lab])

    #ROC-AUC scores can be calculated by binarizing the data
    auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
                        y_prob[lab],
                        average='weighted')
    
    #Last, the confusion matrix
    cm[lab] = confusion_matrix(y_test, y_pred[lab])

    metrics.append(pd.Series({'precision': precision, 'recall': recall,
                              'fscore': fscore, 'accuracy': accuracy,
                              'auc': auc},
                              name=lab))
    

metrics = pd.concat(metrics, axis = 1)

metrics


# %%
#Question 8
#Display or plot the confusion amtrix for each model
fig, axList = plt.subplots(nrows=2, ncols=2)
axList = axList.flatten()
fig.set_size_inches(12, 10)

axList[-1].axis('off')

for ax, lab in zip(axList[:-1], coeff_labels):
    sns.heatmap(cm[lab], ax = ax, annot = True, fmt='d')
    ax.set(title=lab)

plt.tight_layout()

# %%
