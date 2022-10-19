#!/usr/bin/env python
# coding: utf-8

# **Problem Statement :**  
# Our Aim is to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status.

# **Dataset Information**.
# 1) id: unique identifier  
# 2) gender: "Male", "Female" or "Other"  
# 3) age: age of the patient  
# 4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension  
# 5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease  
# 6) ever_married: "No" or "Yes"  
# 7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"  
# 8) Residence_type: "Rural" or "Urban"  
# 9) avg_glucose_level: average glucose level in blood  
# 10) bmi: body mass index  
# 11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"     
# 12) stroke: 1 if the patient had a stroke or 0 if not  
# 
# reference  
# https://www.kaggle.com/code/nikunjmalpani/stroke-prediction-step-by-step-guide/notebook  
# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

# In[31]:


pip install imblearn


# In[32]:


#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import warnings # For suppressing warnings
warnings.filterwarnings("ignore")


# ## Step 1) Importing the data

# In[33]:


# Reading the dataset
df = pd.read_csv(r"C:\Users\Acer\Desktop\data science project\stroke\healthcare-dataset-stroke-data.csv")


# In[34]:


# checking first 5 rows of dataset
print(df.head())


# ## Step2) Data Exploration

# In[35]:


# checking the shape of dataset
print(df.shape)


# **Conclusion**: There are 5110 rows and 12 columns in the dataset.

# In[36]:


# checking missing values
print(df.isna().sum())


# **Conclusion:** The dataset has 1 column of missing value on bmi column

# In[37]:


# Checking any null values, columns data types, numver of entries
print(df.info())


# **Conclusion:**
# The dataset has 5110 total entries, 12 total columns, Only bmi column has missing values. There are 3 floating datatypes, 4 integer datatypes and 5 are the object datatypes.

# In[38]:


# Finding duplicate Rows
df[df.duplicated()].sum()


# **Conclusion:** There are no duplicate rows present in the dataset

# **3. Checking for outliers**

# In[39]:


# Checking for Possible Outliers
fig, axs = plt.subplots(5, figsize = (7,7))
plt1 = sns.boxplot(df['age'], ax = axs[0])
plt2 = sns.boxplot(df['hypertension'], ax = axs[1])
plt3 = sns.boxplot(df['heart_disease'], ax = axs[2])
plt4 = sns.boxplot(df['avg_glucose_level'], ax = axs[3])
plt5 = sns.boxplot(df['bmi'], ax = axs[4])
plt.tight_layout()


# **Conclusion:** The boxplots show that there are huge outliers in bmi and avg_glucose_level

# In[40]:


# reading some statistical information of all the numerical features.
df.describe()


# In[41]:


# checking numbers of stroke
df['stroke'].value_counts()


# **Conclusion:** This dataset was highly unbalanced.

# ## Exploratory Data Analysis

# **1. Distribution of the target variable**

# Checking dataset target

# In[44]:


# Analysis of Stroke - Target Variable
plt.figure(figsize=(8,6))
sns.countplot(x = 'stroke', data = df)
plt.show()


# As we can see that from the above plot, its a class imbalancing problem. The number of people who actually had a stroke are very less in our dataset. We'll use oversampling technique to deal with this.

# In[45]:


sns.countplot(data=df,x='gender',hue='stroke');


# **3. Heatmap**

# In[46]:


# Correlation plot
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)


# **Conclusion: The most correlated features are:**
# 
# No strong correlation is observed between any of the features.
# Variables that are showing some effective correlation are:  
# age, hypertension, heart_disease, ever_married, avg_glucose_level.

# ## Step 3) Data Preprocessing

# In[47]:


# Impute missing values for numerical data
df['bmi'].fillna(df['bmi'].median(),inplace=True)


# In[48]:


# drop irrelavant column
# Dropping ID column as it is unique number assigned to every patient and did not have other meaningful information
df=df.drop('id',axis=1)


# In[49]:


# dropping 'Other' in gender
df.drop(df[df['gender'] == 'Other'].index, inplace = True)
df['gender'].unique()


# In[50]:


outlier_columns = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]

for x in outlier_columns:
    Q1 = df[x].quantile(0.25)
    Q3 = df[x].quantile(0.75)
    IQR=Q3-Q1
    lower_limit= Q1-(1.5 * IQR)
    upper_limit = Q3+(1.5 * IQR)
    df=df[~((df[x]<lower_limit)|(df[x]>upper_limit))]

# checking outlier
fig, axs = plt.subplots(5, figsize = (7,7))
plt1 = sns.boxplot(df['age'], ax = axs[0])
plt2 = sns.boxplot(df['hypertension'], ax = axs[1])
plt3 = sns.boxplot(df['heart_disease'], ax = axs[2])
plt4 = sns.boxplot(df['avg_glucose_level'], ax = axs[3])
plt5 = sns.boxplot(df['bmi'], ax = axs[4])
plt.tight_layout()


# In[69]:


# Separating dataset into input and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[81]:


# categorical data into numeric ones using Label Encoder
categorical_data=df.select_dtypes(include=['object']).columns
le=LabelEncoder()
df[categorical_data]=df[categorical_data].apply(le.fit_transform)
print(df.head())


# In[71]:


#Splitting the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# The dataset is imbalanced. So let’s balance our data. We are going to use SMOTE method for this. It will populate our data with records similar to our minor class. .

# In[72]:


# oversampling the train datsets using SMOTE
smote=SMOTE()
X_train,y_train=smote.fit_resample(X_train,y_train)
#X_test,y_test=smote.fit_resample(X_test,y_test)


# In[73]:


# checking numbers of stroke
# df['stroke'].value_counts()
y_train.value_counts()


# ## Model Building

# In[74]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train) # training (learning step)
y_pred = model.predict(X_test) # testing (examination)


# In[75]:


# Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[76]:


# Precision
from sklearn.metrics import precision_score
precision_score(y_test, y_pred)


# In[77]:


# Recall
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)


# In[80]:


#Confusion matrix and classification report
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)

sns.heatmap(con_mat, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

print(classification_report(y_test, y_pred))


# In[ ]:




