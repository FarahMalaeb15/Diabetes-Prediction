#FARAH MALAEB

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

                                            #---------------------PART1---------------------#

df = pd.read_csv('data_week6.csv') #reading the csv file using pandas

#1-Plot a histogram showing the distribution for the following features present in the dataset
columns1 = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'HbA1c_level', 'diabetes']
df[columns1].hist(color='red', bins=20, figsize=(10, 8))

plt.suptitle('Distribution of Various Features', y=1.02, fontsize=16) # Adding titles and labels
#plt.show() #plotting the grapgh

#2-Plot four different histograms showing the combined distribution of the following two features
columns2 = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle(f' Distribution of with Diabetes', y=1.02, fontsize=16)
# Plot histograms for each combination using a loop
for i in range(len(columns2)): #creating a for loop to make the plotting easier
    col = columns2[i]
    ax = axes.flatten()[i] #groupby expects a single axis to plot on so i used the .flatten to convert the 2D array of subplots (axes) into a 1D array.
    df.groupby(['diabetes', col]).size().unstack().plot(kind='bar', stacked=True, ax=ax) #unstack() reshapes the result into a DataFrame,where one level of the index becomes column
                                                                                          #so here it makes the 'diabetes' values ('0' and '1') separate columns.
                                                                                          #making the grapgh easier to read
    ax.set_title(f'Diabetes and {col}')
    ax.set_xlabel('Diabetes')
    ax.set_ylabel('Frequency')
#plt.show()

#3-Creating a new feature named ‘initial_diagnosis’
df['initial_diagnosis'] = pd.cut(df['HbA1c_level'], bins=[-float('inf'), 5.7, 6.4, float('inf')], labels=['Normal', 'Prediabetes', 'Diabetes'])
plt.figure(figsize=(8, 6))
sns.countplot(x='initial_diagnosis', data=df)
plt.title('Distribution of Initial Diagnosis')
plt.xlabel('Initial Diagnosis')
plt.ylabel('Count')
#plt.show()


##4- plotting three different histograms showing each of the initial_diagnosis categories v.s. if the patient is diabetic or not.
plt.figure(figsize=(12, 8))
for diagnosis_category in df['initial_diagnosis'].unique():
    initialDiag_df = df[df['initial_diagnosis'] == diagnosis_category]
    sns.histplot(initialDiag_df['diabetes'], bins=[-0.5, 0.5, 1.5], stat='count', kde=False, label=diagnosis_category)

plt.title('Distribution of Diabetes for Each Initial Diagnosis Category')
plt.xlabel('Diabetes (0: No, 1: Yes)')
plt.ylabel('Count')
plt.legend(title='Initial Diagnosis')
#plt.show()

#5-Encode the gender and smoking_history columns
print("\nBefore one-hot encoding:")
print(df.columns)
df_encoded = pd.get_dummies(df, columns=['gender', 'smoking_history'])
# After one-hot encoding
print("\nAfter one-hot encoding:")
print(df_encoded.columns)

#6- Check for duplicates
duplicates = df[df.duplicated()]
print('Duplicates', duplicates.shape[0]) #shape[0] refers to the number of rows in the DataFrame 'duplicates' and it represents the count of duplicated rows in the DataFrame.

#7- Dropping the duplicates
dropping_duplicates = df.drop_duplicates() #dropping the duplicates using the drop_duplicates() method
print(f"DataFrame with duplicates: {df.shape}")
print(f"DataFrame  after dropping the duplicates: {dropping_duplicates.shape}")

#Scale all the numeric features using the normalization technique.
#To do that we must use the MinMaxScaler() from the scikit-learn library

from sklearn.preprocessing import MinMaxScaler

numeric_cols = df.select_dtypes(include=['float64']).columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
binary_cols = ['hypertension', 'heart_disease', 'diabetes'] #these columns where binary (0 or 1) before normalization.
df[binary_cols] = df[binary_cols].astype(int) #converting these columns back to integers after normalization, assuming they were originally binary.
print(df.head())
print(df.info())


                                             #---------------------PART2---------------------#

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

features = ['gender_Female', 'smoking_history_current', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level' ,'blood_glucose_level']
X = df_encoded[features] #using the encoded data | X are the features we want to use to predict if the patient has diabetes or not
y = df_encoded['diabetes'] #y is the target variable we want to predict, in this case its diabetes

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42) #splitting the dataset into 80% training and 20% testing.(test_size=o.2 means testing is 20%)
model = LogisticRegression(max_iter=1000) #Creating a logistic regression model
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) #computing the accuracy.
print(f"Accuracy on the testing dataset: {accuracy:.2%}") #Evaluating the performance of the model


new_patient = pd.DataFrame({  #creating a new patient and passing a new data to predict if she has diabetes or no (1 for yes and 0 for no)
    'gender_Female': [1],
    'smoking_history_current': [1],
    'age': [48],
    'hypertension': [0],
    'heart_disease': [1],
    'bmi': [28.4],
    'HbA1c_level': [6.2],
    'blood_glucose_level': [120]

})

prediction = model.predict(new_patient)
print(f"The prediction for the new patient is:", prediction) #printing the prediction of the new patient

if prediction == 0:
    print(f"The prediction is 0 this means that the patient doesn't have diabetes")
else:
    print(f"The prediction is 1 this means that the patient have diabetes")

