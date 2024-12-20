# XGBoost

# Importing the libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd  # For data manipulation and analysis

# Importing the dataset
dataset = pd.read_csv(r"D:\FSDS Material\Dataset\Classification\Churn_Modelling.csv") 
x = dataset.iloc[:, 3:-1].values  # Extracting independent variables (features) from columns 3 to the second last column
y = dataset.iloc[:, -1].values    # Extracting the target variable (Churn)


# Displaying the initial extracted feature set and target variable
print(x)  # Printing the feature set
print(y)  # Printing the target variable

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
le = LabelEncoder()  # Initializing the label encoder
x[:, 2] = le.fit_transform(x[:, 2])  # Encoding the "Gender" column (Male/Female -> 0/1)
print(x)  # Printing the dataset after label encoding

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer  # For applying transformations to specific columns
from sklearn.preprocessing import OneHotEncoder  # For one-hot encoding categorical variables
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')  # Applying one-hot encoding to column 1 ("Geography")
x = np.array(ct.fit_transform(x))  # Converting the transformed data back to a NumPy array
print(x)  # Printing the dataset after one-hot encoding

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  # For splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # 80% training, 20% testing split

# Training XGBoost on the Training set
from xgboost import XGBClassifier  # Importing the XGBoost classifier
classifier = XGBClassifier()  # Initializing the classifier with default parameters
classifier.fit(x_train, y_train)  # Training the model on the training data

# Predicting the Test set results
y_pred = classifier.predict(x_test)  # Making predictions on the test data

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix  # For evaluating the classification model
cm = confusion_matrix(y_test, y_pred)  # Generating the confusion matrix
print(cm)  # Printing the confusion matrix

# Calculating the accuracy score
from sklearn.metrics import accuracy_score  # For calculating the accuracy
ac = accuracy_score(y_test, y_pred)  # Computing the accuracy of the model
print(ac)  # Printing the accuracy

# Calculating the model's performance on the training data (bias)
bias = classifier.score(x_train, y_train)  # Evaluating the model's accuracy on the training set
bias  # Printing the training accuracy (bias)

variance = classifier.score(x_test, y_test)
variance
