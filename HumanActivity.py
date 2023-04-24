# %%
#The data base used for this is the Human Activity Recgoniyion with Smartphones database
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

# %%
import os
import seaborn as sns
import pandas as pd
import numpy as np

# Question 1
    #Import the data and do the following
    #Examine the data if the floating point values need to be scaled 
    #Determine the breakdown of each activity
    #Encode the activity label as an integer

# %%
print('\nPreparing to read Human Activity Recognition\n')
human_activity_data = pd.read_csv('Human_Activity_Recognition.csv')
print('\nData set download check\n', human_activity_data.head(10))