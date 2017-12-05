import numpy as np

# =========================================
# CLASSIFICATION and REGRESSION TREE (CART)
# =========================================

def Node(object):
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

def Leaf(object):
    """
        End of the decision tree, contain the prediction
        Args:
            feature (int): index of the feature
            value (float|string): the class prediction
    """
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

def DecisionTree(object):
    """
        Based DecisionTree for CART trees
        Args:
            max_depth (int): maximum depth of the tree (-1 == no limit)
    """
    def __init__(self, max_depth=-1):
        self.max_depth = max_depth
        
        self._tree = None

    """
        The cost measure used to evaluate the quality of a proposed split.
        Depends on the goal of the three: regression or classification
        Returns:
            float
    """
    def cost(self):
        raise NotImplementedError("cost needs to be implemented")

    """
        Train the decision tree
        Args:
            x (float[][]): normalized training dataset
            y (float[][]): normalized target classes
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
            y (float[][]): normalized target classes
            current_depth (int): current depth of the three
        Returns:
            Node|Leaf
    """
    def _build_decision_tree(x, y, current_depth):
        n_samples, n_features = x.shape

        # When walking the three, check if the node is worth splitting
        #if current_depth < self.max_depth:



        

