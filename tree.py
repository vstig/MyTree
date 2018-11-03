from node import Node


class Tree(object):
    def __init__(self, depth=0, min_samples_leaf=5, max_depth=10):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

        self.depth = depth

        self.root_node = None

        self.left_child = None
        self.right_child = None

        self.is_fitted = False

    def fit(self, X, y):
        if self.is_fitted:
            return "Tree has aleady been fitted"
        self.root_node = Node()
        X_i, X_j = self.root_node.split_data(X, y)
        if (
                X_i is not None and  # this means X is homogeneous, no possible splits
                self.depth + 1 <= self.max_depth and  # have we reached maximum depth?
                X_i.shape[0] >= self.min_samples_leaf and  # would split make left tree have less than min_samples_leaf?
                X_j.shape[0] >= self.min_samples_leaf  # would split make right tree have less than min_samples_leaf?
        ):
            self.left_child = Tree(depth=self.depth + 1,
                                             min_samples_leaf=self.min_samples_leaf,
                                             max_depth=self.max_depth)
            self.left_child.fit(X_i, y)

            self.right_child = Tree(depth=self.depth + 1,
                                              min_samples_leaf=self.min_samples_leaf,
                                              max_depth=self.max_depth)
            self.right_child.fit(X_j, y)
        else:
            self.root_node.is_terminal = True

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            return "Tree not fit yet"
        else:
            node = self.root_node
            while not node.is_terminal:
                if node.which_branch(X) == 'left':
                    node = node.left_child
                else:
                    node = node.right_child

            return node.avg

    def get_decision_path(self, X):
        if not self.is_fitted:
            return "Tree not fit yet"
        else:
            node = self.root_node
            last_y = node.avg
            print('Population average y: {}'.format(node.avg))
            while not node.is_terminal:
                if node.which_branch(X) == 'left':
                    if isinstance(node.feature_value, float):
                        print('{} < {}'.format(node.feature_split, node.feature_value))
                    else:
                        print('{} != {}'.format(node.feature_split, node.feature_value))
                    node = node.left_child
                else:
                    if isinstance(node.feature_value, float):
                        print('{} >= {}'.format(node.feature_split, node.feature_value))
                    else:
                        print('{} == {}'.format(node.feature_split, node.feature_value))
                    node = node.right_child
                diff = node.avg - last_y
                print('\t{} ({}{:.3f})'.format(node.avg, '+' if diff > 0 else '', diff))
                last_y = node.avg