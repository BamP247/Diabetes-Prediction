import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the csv file
df = pd.read_csv("diabetes.csv")

print(df.head(10))

# Select independent and dependent variable
X = df[["Pregnancies",  "Glucose",  "BloodPressure"]]
y = df["Outcome"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Predict the model
pred=classifier.predict(X_test)
pred
# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))

