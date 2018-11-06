import numpy as np


def get_decision_path(tree, X):
    if not tree.is_fitted:
        return "Tree not fit yet"
    else:
        last_y = tree.avg
        print('# Samples: {}\nPopulation average y: {:.3}\n'.format(tree.n_samples, tree.avg))
        branch = tree
        while not branch.is_leaf:
            node = branch.split_node
            if node.which_branch(X) == 'left':
                if np.issubdtype(type(X[node.feature_split]), np.number):
                    print('{} < {}'.format(node.feature_split, node.feature_value))
                else:
                    print('{} != {}'.format(node.feature_split, node.feature_value))

                branch = branch.left_child
            else:
                if np.issubdtype(type(X[node.feature_split]), np.number):
                    print('{} >= {}'.format(node.feature_split, node.feature_value))
                else:
                    print('{} == {}'.format(node.feature_split, node.feature_value))

                branch = branch.right_child

            diff = branch.avg - last_y
            print(('\t||\n\t||\n\t\/\n'
                  '# Samples: {}\tY: {:.3f} ({}{:.2f})\n'
                  ).format(branch.n_samples, branch.avg, '+' if diff > 0 else '', diff))

            last_y = branch.avg
