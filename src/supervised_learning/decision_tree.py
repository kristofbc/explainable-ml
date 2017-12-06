import numpy as np

# =========================================
# CLASSIFICATION and REGRESSION TREE (CART)
# =========================================

class Node(object):
    """
        A decision tree is built with Nodes, which contain other nodes,
        the attibute and value to evaluate.
        Args:
            feature (int): index of the feature
            pivot (float): the value compared when walking the tree
            left (Node): left decision node
            right (Node): right decision node
    """
    def __init__(self, feature, pivot, left, right):
        self.feature = feature
        self.pivot = pivot
        self.left = left
        self.right = right

class Leaf(object):
    """
        End of the decision tree, contain the prediction
        Args:
            value (float|string): the class prediction
    """
    def __init__(self, value):
        self.value = value

class BestSplit(object):
    """
        Simple class holding the information about the best split found
        Args:
            feature (int): index of the feature
            pivot (float|string): threshold of the split
            split_left_x (float[][]|string[][]): left split of the dataset
            split_left_y (float[]|string[]): left split of the label 
            split_right_x (float[][]|string[][]): right split of the dataset 
            split_right_y (float[]|string[]): right split of the label
    """
    def __init__(self, feature, pivot, split_left_x, split_left_y, split_right_x, split_right_y):
        self.feature = feature
        self.pivot = pivot
        self.split_left_x = split_left_x
        self.split_left_y = split_left_y
        self.split_right_x = split_right_x
        self.split_right_y = split_right_y

class DecisionTree(object):
    """
        Based DecisionTree for CART trees
        Args:
            max_depth (int): maximum depth of the tree (-1 == no limit)
            gain_function (function): callback when gain is called
            leaf_function (function): callback when leaf is called
            worth_splitting_function (function): callback when worth_splitting is called
            predict_function (function): callback when predict is called
    """
    def __init__(self, max_depth=-1, gain_function=None, leaf_function=None, worth_splitting_function=None, predict_function=None):
        self.max_depth = max_depth
        self._gain_function = gain_function
        self._leaf_function = leaf_function
        self._worth_splitting_function = worth_splitting_function
        self._predict_function = predict_function
        self._tree = None

    """
        The gain is used to evaluate the quality of a proposed split.
        Depends on the goal of the three: regression or classification
        Args:
            y (float[]): normalized target classes
            split_left_y (float[]): target classed split left
            split_left_y (float[]): target classed split right
        Returns:
            float
    """
    def gain(self, y, split_left_y, split_right_y):
        if self._gain_function is not None:
            return self._gain_function(y, split_left_y, split_right_y)

        raise NotImplementedError("gain needs to be implemented")

    """
        Return the value of the leaf depending on the given labels
        Args:
            y (float[]): labels to get the value from
        Returns:
            float|string
    """
    def leaf(self, y):
        if self._leaf_function is not None:
            return self._leaf_function(y)

        raise NotImplementedError("leaf needs to be implemented")


    """
        Check if the tree should be splitted further, or produce a leaf
        Args:
            current_depth (int): current depth of the tree
            largest_cost (float): cost representing the best split
            largest_cost_set (BestSplit): an instance of BestSplit
        Returns:
            boolean
    """
    def worth_splitting(self, current_depth, largest_cost, largest_cost_set):
        if self._worth_splitting_function is not None:
            return self._worth_splitting_function(current_depth, largest_cost, largest_cost_set)

        raise NotImplementedError("worth_splitting needs to be implemented")

    """
        Train the decision tree
        Args:
            x (float[][]): normalized training dataset
            y (float[]): normalized target classes
    """
    def train(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        self._tree = self._build_decision_three(x, y, current_depth = 0)

    """
        Predict/classify the samples
        Args:
            x (float[][]): normalized sampled to predict/classify
        Returns:
            string[]|float[]
    """
    def predict(self, x):
        if self._tree is None:
            raise RuntimeError("DecisionTree needs to be trained before prediction")

    """
        Build the decision three recursively
        Args:
            x (float[][]): normalized training dataset
            y (float[]): normalized target classes
            current_depth (int): current depth of the three
        Returns:
            Node|Leaf
    """
    def _build_decision_tree(x, y, current_depth):
        n_samples, n_features = x.shape

        # When walking the three, check if the node is worth splitting
        if current_depth < self.max_depth:
            # Walk the dataset
            checked = {}
            largest_gain = 0
            best_split = {}

            for i in range(n_samples):
                for j in range(n_features):
                    # The pivot is the threshold to check
                    pivot = x[i][j]
                    
                    # Don't check the same feature-pivot more than once
                    feature_pivot_label = "{0}-{1}".format(j, pivot)
                    if feature_pivot_label in checked:
                        continue
                    checked[feature_pivot_label] = True

                    # Split the dataset based on the current feature and value
                    # We must walk all values for a feature to evaluate the best split
                    split_left_x, split_left_y, split_right_x, split_right_y = self._split_feature(x, y, j, pivot)

                    # Finaly, we want the gain that results in an higher information gain
                    gain = self.gain(y, split_left_y, split_right_y)
                    if gain > largest_gain:
                        largest_gain = gain
                        # Simple dictionnary that holds the results of the best set
                        best_split = BestSplit(feature, pivot, split_left_x, split_left_y, split_right_x, split_right_y)

            # Update the depth of the tree since we finished one iteration
            current_depth += 1

            # Evaluate if the tree should continued to be built
            # This is where the recursion happen
            if self.worth_splitting(current_depth, largest_gain, best_split):
                left_node = self._build_decision_three(best_split.split_left_x, best_split.split_left_y, current_depth)
                right_node = self._build_decision_three(best_split.split_right_x, best_split.split_right_y, current_depth)
                return Node(best_set.feature, best_set.pivot, left_node, right_node)

            # We reached the end of the tree, create the leaf
            return Leaf(value=self.leaf(y))

    """
        Split the dataset by value for the specified feature
        Args:
            x (float[][]): normalized training dataset
            y (float[]): normalized target classes
            feature (int): index of the feature
            pivot (float|string): threshold required to split the dataset
        Returns:
            left_x (float[][]|string[][]), left_y (float[]|string[]), right_x (float[][]|string[][]), right_y (float[]|string[])
    """
    def _split_feature(x, y, feature, pivot):
        left, right = [], []
        # Get the right predicate based on the type of the pivot
        predicate = None
        if isinstance(pivot, int) or isinstance(pivot, float):
            predicate = lambda left, right: left >= right
        else:
            predicate = lambda left, right: left == right

        # Execute the split based on the feature and pivot
        n_samples, n_features = x.shape
        x_y = np.concatenate((x, y), axis=1)
        for i in range(n_samples):
            if predicate(x[i][feature], pivot):
                left.append(x_y[i])
            else:
                right.append(x_y[i])

        left = np.asarray(left)
        right = np.asarray(right)
        return left[:, n_samples:], left[:, :n_samples], right[:, n_samples:], right[:, :n_samples] 
