import numpy as np
import copy

from src.utils import data
from src.utils import evaluation

# =====================================
# BASE CLASSES FOR DECISION TREE (base)
# =====================================

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
            minimum_sample_count (int): minimum sample count required for split
    """
    def __init__(self, max_depth=-1, minimum_sample_count=2):
        self.max_depth = max_depth
        self.minimum_sample_count = minimum_sample_count
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
        raise NotImplementedError("gain needs to be implemented")

    """
        Return the value of the leaf depending on the given labels
        Args:
            y (float[]): labels to get the value from
        Returns:
            float|string
    """
    def leaf(self, y):
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

        self._tree = self._build_decision_tree(x, y, current_depth=0)

    """
        Predict/classify the samples
        Args:
            x (float[][]): normalized sampled to predict/classify
            maximum_predicted_value (boolean): instead of returning a dictionnary containing the prediction,
                return a single value
        Returns:
            string[]|float[]
    """
    def predict(self, x, maximum_predicted_value=True):
        if self._tree is None:
            raise RuntimeError("DecisionTree needs to be trained before prediction")

        # Walk the tree and return the value of the leaf
        return np.asarray([self._walk_tree(row, self._tree, maximum_predicted_value) for row in x])

    """
        Walk the tree recursively
        Args:
            x (float[]|string[]): value to infer from
            node (Node): current Node to validate
            maximum_predicted_value (boolean): instead of returning a dictionnary containing the prediction,
                return a single value
        Returns:
            dictionary
    """
    def _walk_tree(self, x, node, maximum_predicted_value):
        # We're at Leaf, return the value
        if isinstance(node, Leaf):
            if maximum_predicted_value:
                best_value = None
                highest_count = -1
                for value, count in node.value.items():
                    if count > highest_count:
                        highest_count = count
                        best_value = value
                return best_value

            return node.value

        # Get the required value for the comparison
        pivot = node.pivot
        feature = node.feature
        predicate = self.comparison_operator(pivot)

        # Check the next node to go to
        next_node = None
        if predicate(x[feature], pivot):
            next_node = node.left
        else:
            next_node = node.right

        return self._walk_tree(x, next_node, maximum_predicted_value)

    """
        Get the operator based on the type of the pivot value
        Args:
            pivot (float|string): infer the operator from the pivot value type
        Returns:
            lamda
    """
    def comparison_operator(self, pivot):
        if isinstance(pivot, int) or isinstance(pivot, float):
            return lambda left, right: left >= right
        else:
            return lambda left, right: left == right

    """
        Build the decision three recursively
        Args:
            x (float[][]): normalized training dataset
            y (float[]): normalized target classes
            current_depth (int): current depth of the three
        Returns:
            Node|Leaf
    """
    def _build_decision_tree(self, x, y, current_depth):
        n_samples, n_features = x.shape

        # When walking the three, check if the node is worth splitting
        if n_samples > self.minimum_sample_count and (self.max_depth == -1 or current_depth < self.max_depth):
            # Walk the dataset
            largest_gain = 0
            best_split = {}

            for i in range(n_samples):
                # Don't check the same value for an attribute more than once
                checked = {}

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
                    if len(split_left_x) == 0 or len(split_left_y) == 0 or len(split_right_x) == 0 or len(split_right_y) == 0:
                        continue

                    # Finaly, we want the gain that results in an higher information gain
                    gain = self.gain(y, split_left_y, split_right_y)
                    if gain > largest_gain:
                        largest_gain = gain
                        # Simple dictionnary that holds the results of the best set
                        best_split = BestSplit(j, pivot, split_left_x, split_left_y, split_right_x, split_right_y)

            # Update the depth of the tree since we finished one iteration
            current_depth += 1

            # Evaluate if the tree should continued to be built
            # This is where the recursion happen
            if self.worth_splitting(current_depth, largest_gain, best_split):
                left_node = self._build_decision_tree(best_split.split_left_x, best_split.split_left_y, current_depth)
                right_node = self._build_decision_tree(best_split.split_right_x, best_split.split_right_y, current_depth)
                return Node(best_split.feature, best_split.pivot, left_node, right_node)

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
    def _split_feature(self, x, y, feature, pivot):
        left, right = [], []
        # Get the right predicate based on the type of the pivot
        predicate = self.comparison_operator(pivot)

        # Execute the split based on the feature and pivot
        n_samples, n_features = x.shape
        x_y = np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)
        for i in range(n_samples):
            if predicate(x[i][feature], pivot):
                left.append(x_y[i])
            else:
                right.append(x_y[i])

        # Split the array to the corresponding size and type
        left = np.asarray(left)
        right = np.asarray(right)
        left_x, left_y, right_x, right_y = [], [], [], []
        if len(left) > 0:
            left_x = left[:, :n_features]
            left_y = np.squeeze(left[:, n_features:], axis=1)
        if len(right) > 0:
            right_x = right[:, :n_features]
            right_y = np.squeeze(right[:, n_features:], axis=1)
            
        return left_x, left_y, right_x, right_y

    """
        Prune the tree to prevent overfitting via cross-validation on gain
        This function returns a new instance of the DecisionTree, but pruned
        Args:
            alpha (float): minimum gain to achieve
        Returns:
            DecisionTree
    """
    def prune(self, alpha=0.5):
        if self._tree is None:
            raise RuntimeError("tree should be trained before pruning")

        # Recursively walk the tree
        pruned_inst = copy.deepcopy(self)
        pruned_inst._tree = pruned_inst._prune_tree(pruned_inst._tree, alpha)
        return pruned_inst

    """
        Walk the tree recursively and prune it based on the gain and alpha value
        Args:
            node (Node|Leaf): current node to evaluate
            alpha (float): minimum accepted gain for pruning
        Returns:
            Node|Leaf
    """
    def _prune_tree(self, node, alpha):
        # We reach the bottom, return the Leaf
        if isinstance(node, Leaf):
            return node

        # Walk the tree and find a Lead node if possible
        if not isinstance(node.left, Leaf):
            node.left = self._prune_tree(node.left, alpha)
        if not isinstance(node.right, Leaf):
            node.right = self._prune_tree(node.right, alpha)
        
        # Merge leaves based on the gain and alpha
        if isinstance(node.left, Leaf) and isinstance(node.right, Leaf):
            left, right = [], []
            # Create a list of values expanded count times
            left = [[value] * int(count) for value, count in node.left.value.items()]
            right = [[value] * int(count) for value, count in node.right.value.items()]
            if len(left) > 0:
                left = np.concatenate(left, axis=0)
            if len(right) > 0:
                right = np.concatenate(right, axis=0)

            # Get the gain on our new "y", which is the concatenation of left and right
            y = np.concatenate((left, right), axis=0)
            gain = self.gain(y, left, right)

            # Prune if the gain is bellow the threshold
            if gain < alpha:
                print("Node was pruned")
                # The Node becomes a Leaf
                return Leaf(value=self.leaf(y))

        # Return the normal Node, no merge was performed
        return node
        

class FunctionnalDecisionTree(DecisionTree):
    """
        DecisionTree using arbitrary functions for missing implementation
        Usefull when testing different gain or other functions
        Args:
            max_depth (int): maximum depth of the tree (-1 == no limit)
            minimum_sample_count (int): minimum sample count required for split
            gain_function (function): callback when gain is called
            leaf_function (function): callback when leaf is called
            worth_splitting_function (function): callback when worth_splitting is called
    """
    def __init__(self, max_depth=-1, minimum_sample_count=2, gain_function=None, leaf_function=None, worth_splitting_function=None):
        super(FunctionnalDecisionTree, self).__init__(max_depth, minimum_sample_count)

        self._gain_function = gain_function
        self._leaf_function = leaf_function
        self._worth_splitting_function = worth_splitting_function

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

        return super(FunctionnalDecisionTree, self).gain(y, split_left_y, split_right_y)

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

        return super(FunctionnalDecisionTree, self).leaf(y)


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

        return super(FunctionnalDecisionTree, self).worth_splitting(current_depth, largest_cost, largest_cost_set)

    """
        Set the gain function
        Args:
            f (function): the function to use as the gain function
    """
    def set_gain_function(self, f):
        self._gain_function = f

    """
        Set the leaf function
        Args:
            f (function): the function to use as the leaf function
    """
    def set_leaf_function(self, f):
        self._leaf_function = f

    """
        Set the worth splitting function
        Args:
            f (function): the function to use as the worth_splitting function
    """
    def set_worth_splitting_function(self, f):
        self._worth_splitting_function = f

# Generic functions that can be used with FunctionnalDecisionTree
"""
    Used as a gain function, it takes a generic cost function to compute the normalized measure

    gain = cost_function(y) - (len(y_left)/len(y) * cost_function(y_left) + len(y_right)/len(y) * cost_function(y_right))

    Args:
        cost_function (function): a function returning the cost of y
    Returns:
        float
"""
def normalized_measure_reduction_cost(cost_function):
    def gain(y, split_left_y, split_right_y):

        p = len(split_left_y) - len(y)
        y_cost = cost_function(y)
        y1_cost = cost_function(split_left_y)
        y2_cost = cost_function(split_right_y)

        gain = y_cost - p*y1_cost - (1.0-p)*y2_cost
        return gain

    return gain

"""
    Entropy or deviance is the measure of "impurity"
    It quantifies the homogeneity of the dataset (homegeneous = 1, perfect 50% = 0)

    cost = -p(a)*log(p(a)) - p(b)*log(p(b))

    Args:
        y (float[]|string[]): entropy is computed on these values
    Returns:
        float
"""
def cost_function_entropy(y):
    log2 = lambda x: np.log(x) / np.log(2.0)
    unique = data.count_unique_value(y)
    entropy = 0.0
    for value, count in unique.items():
        p = count/len(y)
        entropy += -p*log2(p)
    return entropy

"""
    Variance of y

    cost = \sum(y_i - mean(y))^2

    Args:
        y (float[]|string[]): sum-of-square is computed on these values
    Returns:
        float
"""
def cost_function_variance(y):
    y_mean = np.mean(y)
    return np.sum(np.square(y - y_mean))

"""
    The gini index is the expected error rate

    cost = \sum(len(y_k)/len(y) * (1 - (len(y_k)/len(y))))

    Args:
        y (float[]|string[]): sum-of-square is computed on these values
    Returns:
        float
"""
def cost_function_gini(y):
    unique = data.count_unique_value(y)
    gini = 0.0
    for value1, count1 in unique.items():
        p1 = count1/len(y)
        for value2, count2 in unique.items():
            if value1 == value2:
                continue
            p2 = count2/len(y)
            gini += p1*p2

    return gini

# =================================================================
# IMPLEMENTATION OF CLASSIFICATION and REGRESSION TREE: CART (impl)
# =================================================================
def ClassificationDecisionTree(minimum_reduction_cost=1e-7, max_depth=-1, minimum_sample_count=2, cost_function=cost_function_entropy):
    def leaf(y):
        # Count how many time a value is found inside the data
        # {value: count}
        return data.count_unique_value(y)

    def worth_splitting(current_depth, largest_cost, largest_cost_set):
        return largest_cost > minimum_reduction_cost

    return FunctionnalDecisionTree(max_depth, minimum_sample_count, 
                                   gain_function=normalized_measure_reduction_cost(cost_function), 
                                   leaf_function=leaf, worth_splitting_function=worth_splitting)

def RegressionDecisionTree(minimum_reduction_cost=1e-7, max_depth=-1, minimum_sample_count=2, cost_function=cost_function_variance):
    def leaf(y):
        # Leaf contain the mean of y
        value = np.mean(y)
        return {value: 1.0}

    def worth_splitting(current_depth, largest_cost, largest_cost_set):
        return largest_cost > minimum_reduction_cost

    return FunctionnalDecisionTree(max_depth, minimum_sample_count, 
                                   gain_function=normalized_measure_reduction_cost(cost_function), 
                                   leaf_function=leaf, worth_splitting_function=worth_splitting)

