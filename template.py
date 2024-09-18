import pdt_lib as pdt

def calculate_metrics(actual_labels, predicted_labels):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, average='macro')
    recall = recall_score(actual_labels, predicted_labels, average='macro')
    f1 = f1_score(actual_labels, predicted_labels, average='macro')

    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

# Import necessary libraries
import sklearn.datasets  # To load the Iris dataset
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets

# Load the Iris dataset from sklearn
iris = sklearn.datasets.load_iris()
data = iris['data']  # Features (input data)
labels = iris['target']  # Labels (output data)

# Split the data into training and testing sets
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the decision tree using your custom PDT library
tree = pdt.build_pdt(data_train, label_train, 0, 100)

# Make predictions on the test data using the built decision tree
predicted_labels_test = pdt.prediction(data_test, tree)

# Make predictions on the training data as well
predicted_labels_train = pdt.prediction(data_train, tree)


# Print and calculate metrics for the test data
print("This is for test data")
calculate_metrics(label_test, predicted_labels_test)

# Print and calculate metrics for the training data
print("This is for train data")
calculate_metrics(label_train, predicted_labels_train)

# Visualize the decision tree
pdt.plot(tree, "iris")