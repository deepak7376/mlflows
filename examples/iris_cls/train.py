import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the tracking URI
mlflow.set_tracking_uri("https://localhost:5000")

# Start an MLflow run
mlflow.start_run()

# Define and train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Log parameters, metrics, and the model
mlflow.log_params({'n_estimators': 100})
mlflow.log_metric('accuracy', accuracy)
mlflow.sklearn.log_model(model, 'model')

# End the MLflow run
mlflow.end_run()
