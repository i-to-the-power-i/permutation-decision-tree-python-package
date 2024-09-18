# Import necessary libraries
import numpy as np
from graphviz import Digraph
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from collections import Counter

# Function to calculate ETC value for a given sequence
def calculate_etc(seq):
    """
    Calculate the ETC (Effort To Compress) value for a given sequence of numbers.
    
    Args:
    seq (list or string): Input sequence for which ETC needs to be calculated.

    Returns:
    int: The value of ETC (number of iterations to compress the sequence).
    """
    itrs = 0
    seq = list(seq)  # Convert to list if it's not already
    while len(set(seq)) > 1:  # Loop until the sequence is uniform
        # Count pairs of consecutive elements
        pair_count = Counter((seq[i], seq[i + 1]) for i in range(len(seq) - 1))
        
        # Find the most frequent pair
        highest_freq_pair = max(pair_count, key=pair_count.get)
        
        # Create a new symbol that is larger than the current max value in the sequence
        new_symbol = str(max(map(int, seq)) + 1)
        
        # Find indices where the most frequent pair occurs
        indices_to_replace = [i for i in range(len(seq) - 1) if tuple(seq[i:i + 2]) == highest_freq_pair]
        
        # Replace the most frequent pairs with the new symbol
        temp_seq = []
        skip_next = False  # To skip the next element after replacement
        for i in range(len(seq)):
            if skip_next:
                skip_next = False
                continue
            if i in indices_to_replace:
                temp_seq.append(new_symbol)
                skip_next = True  # Skip the next element as it's part of the replaced pair
            else:
                temp_seq.append(seq[i])
        
        # Update the sequence and iteration count
        seq = temp_seq
        itrs += 1

    return itrs

# Function to calculate ETC gain for a specific feature
def etc_gain(data, labels, feature_index, resolution=2):
    """
    Calculate the ETC gain for a specific feature in the dataset.

    Args:
    data (list): Dataset with feature values.
    labels (list): List of labels associated with the data.
    feature_index (int): Index of the feature to calculate the gain for.

    Returns:
    tuple of int: Best gain and threshold for the feature.
    """
    total_etc = calculate_etc(labels)
    
    # Extract the feature values for the given feature_index from the data
    feature_values = [row[feature_index] for row in data]
    
    best_gain = -float('inf')  # Initialize best gain to negative infinity
    best_threshold = None  # Initialize best threshold as None
    
    # Create an array of threshold values to test, ranging between min and max of feature values
    threshold_array = np.linspace(min(feature_values), max(feature_values), resolution * len(feature_values))
    
    # Iterate over each threshold in the threshold array
    for threshold in threshold_array:
        # Split the labels based on whether their corresponding feature value is <= or > the threshold
        left_labels = [labels[x] for x in range(len(feature_values)) if feature_values[x] <= threshold]
        right_labels = [labels[x] for x in range(len(feature_values)) if feature_values[x] > threshold]
        
        # Calculate the ETC for both the left and right splits
        left_etc = calculate_etc(left_labels)
        right_etc = calculate_etc(right_labels)
        
        # Calculate the proportion of data points in each split (weights)
        left_weight = len(left_labels) / len(labels)
        right_weight = len(right_labels) / len(labels)
        
        # Calculate the weighted ETC for the current split
        weighted_etc = left_weight * left_etc + right_weight * right_etc
        
        # Calculate the gain (improvement) by subtracting the weighted ETC from the total ETC
        gain = total_etc - weighted_etc
        
        # If the current gain is better than the best gain, update best gain and best threshold
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    
    # Return the best gain and the corresponding threshold
    return best_gain, best_threshold

# Function to find the best feature for splitting the data
def find_best_feature(data, labels, resolution):
    """
    Find the best feature and corresponding threshold for splitting the data.

    Args:
    data (list): Dataset with feature values.
    labels (list): List of labels associated with the data.

    Returns:
    tuple: Best feature index and threshold value for splitting the data.
    """
    num_features = len(data[0])  # Number of features in the dataset
    best_feature_index = -1  # Initialize the best feature index to an invalid value
    best_threshold = -1  # Initialize the best threshold to an invalid value
    best_gain = -1  # Initialize the best gain as a negative value

    # Iterate over all features in the dataset
    for feature_index in range(num_features):
        # Calculate the gain and threshold for the current feature
        gain, local_threshold = etc_gain(data, labels, feature_index, resolution)
        
        # If the current gain is greater than the best gain found so far, update the best values
        if gain > best_gain:
            best_gain = gain
            best_feature_index = feature_index
            best_threshold = local_threshold
    
    # Return the best feature index and the corresponding threshold
    return best_feature_index, best_threshold

# Function to build the Predictive Decision Tree (PDT)
def build_pdt(data, labels, depth=0, max_depth=10, resolution=2):
    """
    Build the Permutation Decision Tree (PDT) using the given dataset and labels.

    Args:
    data (list): Dataset with feature values.
    labels (list): List of labels associated with the data.
    depth (int): Current depth of the tree (default is 0).
    max_depth (int): Maximum depth of the tree (default is 10).
    resolution (int): The number of thresholds used to split the data.

    Returns:
    dict or label: The decision tree structure or a leaf node label.
    """
    # Base cases for stopping the recursion
    if len(data) == 0:  # If no data, return None
        return None
    if len(set(labels)) == 1:  # If all labels are the same, return the label (leaf node)
        return labels[0]
    if depth >= max_depth:  # If maximum depth is reached, return the most frequent label
        return max(set(labels), key=labels.count)

    # Find the best feature and corresponding threshold for the current split
    best_feature, best_threshold = find_best_feature(data, labels, resolution)

    # Initialize the tree node with the best feature and threshold
    tree = {'feature_index': best_feature, 'threshold': best_threshold, 'children': {}}

    # Split data and labels into left (<= threshold) and right (> threshold) branches
    left_data = [row for row in data if row[best_feature] <= best_threshold]
    left_labels = [labels[i] for i, row in enumerate(data) if row[best_feature] <= best_threshold]
    right_data = [row for row in data if row[best_feature] > best_threshold]
    right_labels = [labels[i] for i, row in enumerate(data) if row[best_feature] > best_threshold]

    # Recursively build the tree for the left and right branches
    tree['children']['left'] = build_pdt(left_data, left_labels, depth + 1, max_depth, resolution)
    tree['children']['right'] = build_pdt(right_data, right_labels, depth + 1, max_depth, resolution)

    # Optimization: Try to merge redundant nodes
    try:
        # If the left child node has the same feature and threshold, merge it
        if (tree['children']['left']['feature_index'] == tree['feature_index'] and
            tree['children']['left']['threshold'] == tree['threshold']):
            tree['children'] = tree['children']['left']['children']
        
        # If the right child node has the same feature and threshold, merge it
        if (tree['children']['right']['feature_index'] == tree['feature_index'] and
            tree['children']['right']['threshold'] == tree['threshold']):
            tree['children'] = tree['children']['right']['children']
    except:
        # If the children are leaf nodes (not dicts), skip merging
        pass

    # Return the constructed tree
    return tree

# Function to predict the label for a given data point
def predict(tree, data_point):
    """
    Predict the label for a given data point using the decision tree.

    Args:
    tree (dict): The decision tree structure.
    data_point (list): A single data point to be classified.

    Returns:
    label: Predicted label for the data point.
    """
    # Base case: if the current node is a leaf node (not a dict), return the label
    if not isinstance(tree, dict):
        return tree

    # Get the feature index and threshold for the current node
    feature_index = tree['feature_index']
    threshold = tree['threshold']

    # Compare the data point's feature value to the threshold
    if str(data_point[feature_index]) <= str(threshold):
        # If the feature value is less than or equal to the threshold, recurse on the left child
        return predict(tree['children']['left'], data_point)
    else:
        # If the feature value is greater than the threshold, recurse on the right child
        return predict(tree['children']['right'], data_point)


# Function to generate and save a visual representation of the decision tree
def generate_tree_visuals(tree, name, graph=None):
    """
    Generate a visual representation of the decision tree and save it to a file.

    Args:
    tree (dict): The decision tree structure.
    name (str): The name of the file where the tree visualization will be saved.
    graph (graphviz.Digraph): Graph object to create the visualization (default is None).

    Returns:
    graphviz.Digraph: The graph object with the tree visualization.
    """
    
    def visualize_tree(tree, node_id=0, graph=None):
        """
        Recursively visualize the decision tree structure using Graphviz.

        Args:
        tree (dict or label): The decision tree structure or a leaf node.
        node_id (int): Unique identifier for the current node in the graph.
        graph (graphviz.Digraph): Graphviz Digraph object to build the tree (default is None).
        """
        # Initialize the graph if not provided
        if graph is None:
            graph = Digraph()

        # If the current node is a decision node, create a node with the feature and threshold
        if 'feature_index' in tree and 'threshold' in tree:
            label = f"f{tree['feature_index']} <= {round(tree['threshold'], 2)}"
            graph.node(str(node_id), label)

            # Get the children of the current node (left and right)
            children = tree.get('children', {})
            left = children.get('left', None)
            
            # Visualize the left child if it's a decision node, or add a leaf node
            if isinstance(left, dict):
                left_id = node_id * 2 + 1
                graph.edge(str(node_id), str(left_id), "True")
                visualize_tree(left, left_id, graph)
            elif left is not None:
                left_id = node_id * 2 + 1
                graph.node(str(left_id), f"{left}")
                graph.edge(str(node_id), str(left_id), "True")

            # Visualize the right child if it's a decision node, or add a leaf node
            right = children.get('right', None)
            if isinstance(right, dict):
                right_id = node_id * 2 + 2
                graph.edge(str(node_id), str(right_id), "False")
                visualize_tree(right, right_id, graph)
            elif right is not None:
                right_id = node_id * 2 + 2
                graph.node(str(right_id), f"{right}")
                graph.edge(str(node_id), str(right_id), "False")

        # If the current node is a leaf, add it as a leaf node
        else:
            label = f"Value: {tree}"
            graph.node(str(node_id), label)

        return graph

    # Create the tree visualization and save it as a PNG file
    graph = visualize_tree(tree)
    graph.render(name, format='png', cleanup=True)


# Function to make predictions on the entire dataset
def prediction(data, tree):
    """
    Make predictions for the entire dataset using the decision tree.

    Args:
    data (list): Dataset with feature values.
    tree (dict): The decision tree structure.

    Returns:
    list: Predicted labels for the dataset.
    """
    y_pred = []
    # Loop through each data point and predict its label using the decision tree
    for x in data:
        out = predict(tree, x)
        y_pred.append(out)
    return y_pred  # Return the list of predicted labels
def plot(tree, name="PDT_Visualization"):
    """
    Generate and display a visual representation of the decision tree.

    Args:
    tree (dict): The decision tree structure.
    name (str): The name of the file where the tree visualization will be saved (default is "PDT_Visualization").

    Returns:
    None
    """
    # Generate the decision tree visual and save it as an image
    generate_tree_visuals(tree, name)
    
    # Load the saved image for display
    img = mpimg.imread(f'{name}.png')
    
    # Create a figure and axis for plotting the image
    fig, ax = plt.subplots()
    
    # Display the image without axes
    ax.imshow(img)
    ax.axis('off')
    
    # Show the image
    plt.show()
