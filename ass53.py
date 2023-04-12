import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the Iris dataset
iris = pd.read_csv("Iris.csv")

# Split the data into features and target variable
X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

# Apply LDA with k=2
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Create a new dataframe with the LDA components and target variable
data_lda = pd.DataFrame(X_lda, columns=["LDA1", "LDA2"])
data_lda["target"] = y

# Print the first 5 rows of the new dataframe
print(data_lda.head())
