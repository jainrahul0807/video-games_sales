import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

# Load the csv file
vgsales = pd.read_csv("vgsales.csv")
vgsales= vgsales.dropna(subset=['Year','Publisher'], axis=0)
vgsales = vgsales.drop_duplicates()
vgsales = vgsales.reset_index(drop=True)

# Calculate the total sales per game
vgsales['Total_Sales'] = vgsales['NA_Sales'] + vgsales['EU_Sales'] + vgsales['JP_Sales'] + vgsales['Other_Sales']

# Drop unnecessary columns
vgsales = vgsales.drop(['Rank', 'Name', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)

# Select independent and dependent variable
features = vgsales[['Platform', 'Genre', 'Publisher', 'Total_Sales']]
cat_cols=['Platform', 'Genre', 'Publisher']
features_fr = pd.DataFrame(features)
target = vgsales['Global_Sales']

preprocessor= ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_cols)], remainder='passthrough')

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(features_fr, target, test_size=0.2, random_state=42)

# Instantiate the model
tree_model = Pipeline([('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor())])

# Fit the model
tree_model.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(tree_model, open("model.pkl", "wb"))

import os
os.getcwd()
