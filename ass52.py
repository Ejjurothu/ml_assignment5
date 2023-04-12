import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# Load the data
data = pd.read_csv("pd_speech_features.csv", header=1)

# Split the data into features and target variable
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Perform PCA with k=3
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Fit the SVM model on the training set
svm = SVC()
svm.fit(X_train, y_train)

# Predict the target variable for the testing set
y_pred = svm.predict(X_test)

# Evaluate the performance of the model on the testing set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
