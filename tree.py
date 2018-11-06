from splitnode import SplitNode
import numpy as np


class Tree(object):
    def __init__(self, depth=0, min_samples_leaf=5, max_depth=10, min_reduction=0):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_reduction = 0

        self.depth = depth

        self.split_node = None

        self.avg = float('inf')
        self.n_samples = 0

        self.left_child = None
        self.right_child = None

        self.is_fitted = False

    def fit(self, X, y):
        if self.is_fitted:
            return "Tree has aleady been fitted"

        self.avg = y.mean()
        self.n_samples = X.shape[0]

        self.split_node = SplitNode()

        best_split = self.split_node.split_data(X, y, min_reduction=self.min_reduction,
                                                min_samples_leaf=self.min_samples_leaf)

        if best_split is None or self.depth > self.max_depth:
            self.split_node.is_terminal = True
            self.is_fitted = True

        else:
            (X_i, y_i), (X_j, y_j) = best_split
            self.left_child = Tree(depth=self.depth + 1,
                                             min_samples_leaf=self.min_samples_leaf,
                                             max_depth=self.max_depth)
            self.left_child.fit(X_i, y_i)

            self.right_child = Tree(depth=self.depth + 1,
                                              min_samples_leaf=self.min_samples_leaf,
                                              max_depth=self.max_depth)
            self.right_child.fit(X_j, y_j)

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            return "Tree not fit yet"
        else:
            branch = self
            while not branch.is_leaf:
                if branch.split_node.which_branch(X) == 'left':
                    branch = branch.left_child
                else:
                    branch = branch.right_child

            return branch.avg

    @property
    def is_leaf(self):
        return self.left_child is None
