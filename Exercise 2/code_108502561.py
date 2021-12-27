import sklearn.metrics
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math
import numpy as np
import pandas as pd

#grade: 100

class Node:
    "Decision tree node"
    def __init__(self, entropy, num_samples, num_samples_per_class, predicted_class, num_errors, alpha=float("inf")):
        self.entropy = entropy # the entropy of current node
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class # the majority class of the split group
        self.feature_index = 0 # the feature index we used to split the node
        self.threshold = 0 # for binary split
        self.left = None # left child node
        self.right = None # right child node
        self.num_errors = num_errors # error after cut
        self.alpha = alpha # each node alpha


class DecisionTreeClassifier:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
    def _entropy(self,sample_y,n_classes):
        # calculate the entropy of sample_y and return it
        # sample_y represent the label of node
        # entropy = -sum(pi * log2(pi))
        entropy = 0
        num_nonzero = np.count_nonzero(sample_y)
        num_zero = sample_y.size - num_nonzero
        pi = num_nonzero / len(sample_y)

        npi = num_zero / len(sample_y)
        if num_nonzero == 0:
            entropy = - (npi * np.log2(npi))
        elif num_zero == 0:
            entropy = - (pi * np.log2(pi))
        else:
            entropy = - (pi * np.log2(pi)) - (npi * np.log2(npi))        
            #print(entropy)
        return entropy

    def predict(self, X):
        #Predict class for X.
        return [self._predict(inputs) for inputs in X]
        
    def _feature_split(self, X, y,n_classes):
        # Returns:
        #  best_idx: Index of the feature for best split, or None if no split is found.
        #  best_thr: Threshold to use for the split, or None if no split is found.
        m = y.size
        if m <= 1:
            return None, None

        # Entropy of current node.

        best_criterion = self._entropy(y,n_classes)

        best_idx, best_thr = None, None
        # TODO: find the best split, loop through all the features, and consider all the
        # midpoints between adjacent training samples as possible thresholds. 
        # Compute the Entropy impurity of the split generated by that particular feature/threshold
        # pair, and return the pair with smallest impurity.
        info_gain = 0
        for idx in range(self.n_features_):
            for j in range(len(X[:, 0])):
                num_left = 0
                num_right = 0
                left = np.array([])
                right = np.array([])
                for k in range(len(X[:, 0])):
                    thr = X[j, idx]
                    if X[k, idx] >= thr:
                        num_right += 1
                        right = np.append(right, y[k])
                        right = np.array(right)
                        #print(right)
                    elif X[k,idx] < thr:
                        num_left += 1
                        left = np.append(left, y[k])
                if num_right == 0:
                    info_en = self._entropy(left, n_classes)
                elif num_left == 0:
                    info_en = self._entropy(right, n_classes)
                else:
                    info_en = (num_left/m)*self._entropy(left, n_classes) + (num_right/m)*self._entropy(right, n_classes)
                Gain = best_criterion - info_en
                #print("best_criterion: ", best_criterion)
                #print("info: ", info_en)
                #print("Gain: ", Gain)
                if Gain > info_gain:
                    info_gain = Gain           
                    best_idx = idx
                    best_thr = thr
                #print("info_gain: ", info_gain)
                #print("best_idx: ", best_idx)
                #print("best_thr: ", best_thr)                    

        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        correct_label_num = num_samples_per_class[predicted_class]
        num_errors = y.size - correct_label_num
        node = Node(
            entropy = self._entropy(y,self.n_classes_),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            num_errors=num_errors
        )

        if depth < self.max_depth:
            idx, thr = self._feature_split(X, y,self.n_classes_)
            if idx is not None:
            #Split the tree recursively according index and threshold until maximum depth is reached.
                node.feature_index = idx
                node.threshold = thr
                indices_left = X[:, idx] < thr
                indices_right = ~indices_left
                #print(" X[indices_left]: " ,X[indices_left])
                #print("X[indices_right}", X[~indices_left])
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[indices_right], y[indices_right]
                node.left = self._build_tree(X_left, y_left, depth + 1)
                node.right = self._build_tree(X_right, y_right, depth + 1)
        return node

    def fit(self,X,Y):
        # Fits to the given training data
        self.n_classes_ = 2
        #print(len(set(Y)))
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X,Y)
        pass

    def _predict(self,inputs):
        #predict the label of data
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def _find_leaves(self, root):
        depth = 0
        while(root.right != None):
            depth += 1
            root = root.right
        ## find each node child leaves number
        # leaf num = depth*2
        return depth*2
    def _error_before_cut(self, root):
        # TODO
        ## return error before post-pruning
        EBC = 0
        stack = []
        stack.append(root)
        while(1):
            if len(stack) == 0:
                break
            
            node = stack.pop()
            if node.left != None and node.right != None:
                stack.append(node.left)
                stack.append(node.right)
            elif node.left == None and node.right == None:
                EBC += node.num_errors
        return EBC

    def _compute_alpha(self, root):
        # TODO
        ## Compute each node alpha
        # alpha = (error after cut - error before cut) / (leaves been cut - 1)
        stack = []
        stack.append(root)
        while(1):
            if len(stack) == 0:
                break
            
            node = stack.pop()
            if node.left != None and node.right != None:
                stack.append(node.left)
                stack.append(node.right)
                node.alpha = (node.num_errors - self._error_before_cut(node)) / (self._find_leaves(node) -1)
            else:
                node.alpha = float("inf")
        pass
    
    def _find_min_alpha(self, root):
        MinAlpha = float("inf")
        ## Search the Decision tree which have minimum alpha's node
        min_node = root
        stack = []
        stack.append(root)
        while (True):
            if len(stack) == 0:
                break

            node = stack.pop()

            if (node.left != None and node.right != None):
                stack.append(node.left)
                stack.append(node.right)
            if (MinAlpha > node.alpha):
                MinAlpha = node.alpha
                min_node = node

        return min_node

    def _prune(self):
        self._compute_alpha(self.tree_)
        cut_node = self._find_min_alpha(self.tree_)

        ## prune the decision tree with minimum alpha node
        cut_node.left = None
        cut_node.right = None
        pass


def load_train_test_data(test_ratio=.3, random_state = 1):
    df = pd.read_csv('.\heart_dataset.csv')
    X = df.drop(columns=['target'])
    X = np.array(X.values)
    y = df['target']
    y = np.array(y.values)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_ratio, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def accuracy_report(X_train_scale, y_train,X_test_scale,y_test,max_depth=7):
    tree = DecisionTreeClassifier( max_depth=max_depth)
    tree.fit(X_train_scale, y_train)
    pred = tree.predict(X_train_scale)

    print(" tree train accuracy: %f" 
        % (sklearn.metrics.accuracy_score(y_train, pred )))
    pred = tree.predict(X_test_scale)
    print(" tree test accuracy: %f" 
        % (sklearn.metrics.accuracy_score(y_test, pred )))
    
    for i in range(10):
        print("=============Cut=============")
        tree._prune()
        pred = tree.predict(X_train_scale)
        print(" tree train accuracy: %f" 
            % (sklearn.metrics.accuracy_score(y_train, pred )))
        pred = tree.predict(X_test_scale)
        print(" tree test accuracy: %f" 
            % (sklearn.metrics.accuracy_score(y_test, pred )))

def main():
    X_train, X_test, y_train, y_test = load_train_test_data(test_ratio=.3,random_state = 1)
    accuracy_report(X_train, y_train,X_test,y_test,max_depth=4)
if __name__ == "__main__":
    main()