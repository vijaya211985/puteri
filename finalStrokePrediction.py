pip install imblearn
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


df = pd.read_csv(r"C:\Users\Acer\Desktop\data science project\stroke\healthcare-dataset-stroke-data.csv")


print(df.head())


print(df.shape)

print(df.isna().sum())

print(df.info())


df[df.duplicated()].sum()


fig, axs = plt.subplots(5, figsize = (7,7))
plt1 = sns.boxplot(df['age'], ax = axs[0])
plt2 = sns.boxplot(df['hypertension'], ax = axs[1])
plt3 = sns.boxplot(df['heart_disease'], ax = axs[2])
plt4 = sns.boxplot(df['avg_glucose_level'], ax = axs[3])
plt5 = sns.boxplot(df['bmi'], ax = axs[4])
plt.tight_layout()


df.describe()

df['stroke'].value_counts()



plt.figure(figsize=(8,6))
sns.countplot(x = 'stroke', data = df)
plt.show()



sns.countplot(data=df,x='gender',hue='stroke');


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)


df['bmi'].fillna(df['bmi'].median(),inplace=True)


df=df.drop('id',axis=1)


df.drop(df[df['gender'] == 'Other'].index, inplace = True)
df['gender'].unique()



outlier_columns = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]

for x in outlier_columns:
    Q1 = df[x].quantile(0.25)
    Q3 = df[x].quantile(0.75)
    IQR=Q3-Q1
    lower_limit= Q1-(1.5 * IQR)
    upper_limit = Q3+(1.5 * IQR)
    df=df[~((df[x]<lower_limit)|(df[x]>upper_limit))]

fig, axs = plt.subplots(5, figsize = (7,7))
plt1 = sns.boxplot(df['age'], ax = axs[0])
plt2 = sns.boxplot(df['hypertension'], ax = axs[1])
plt3 = sns.boxplot(df['heart_disease'], ax = axs[2])
plt4 = sns.boxplot(df['avg_glucose_level'], ax = axs[3])
plt5 = sns.boxplot(df['bmi'], ax = axs[4])
plt.tight_layout()


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

categorical_data=df.select_dtypes(include=['object']).columns
le=LabelEncoder()
df[categorical_data]=df[categorical_data].apply(le.fit_transform)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



smote=SMOTE()
X_train,y_train=smote.fit_resample(X_train,y_train)

y_train.value_counts()




from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train) # training (learning step)
y_pred = model.predict(X_test) # testing (examination)


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


from sklearn.metrics import precision_score
precision_score(y_test, y_pred)


from sklearn.metrics import recall_score
recall_score(y_test, y_pred)


from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)

sns.heatmap(con_mat, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

print(classification_report(y_test, y_pred))






