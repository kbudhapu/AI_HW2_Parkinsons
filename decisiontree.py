# decisiontree.py
"""Predict Parkinson's disease based on dysphonia measurements using a decision tree."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

# ***MODIFY CODE HERE***
ROOT = 'data'  # change to path where data is stored
ROOT = 'data'  # change to path where data is stored
THIS = os.path.dirname(os.path.realpath(__file__))  # the current directory of this file

parser = argparse.ArgumentParser(description="Use a Decision Tree model to predict Parkinson's disease.")
parser.add_argument('-xtrain', '--training_data',
                    help='path to training data file, defaults to ROOT/training_data.txt',
                    default=os.path.join(ROOT, 'training_data.txt'))
parser.add_argument('-ytrain', '--training_labels',
                    help='path to training labels file, defaults to ROOT/training_labels.txt',
                    default=os.path.join(ROOT, 'training_labels.txt'))
parser.add_argument('-xtest', '--testing_data',
                    help='path to testing data file, defaults to ROOT/testing_data.txt',
                    default=os.path.join(ROOT, 'testing_data.txt'))
parser.add_argument('-ytest', '--testing_labels',
                    help='path to testing labels file, defaults to ROOT/testing_labels.txt',
                    default=os.path.join(ROOT, 'testing_labels.txt'))
parser.add_argument('-a', '--attributes',
                    help='path to file containing attributes (features), defaults to ROOT/attributes.txt',
                    default=os.path.join(ROOT, 'attributes.txt'))
parser.add_argument('--debug', action='store_true', help='use pdb.set_trace() at end of program for debugging')
parser.add_argument('--save', action='store_true', help='save tree image to file')
parser.add_argument('--show', action='store_true', help='show tree image while running code')

def main(args):
    print("Training a Decision Tree to Predict Parkinson's Disease")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")

    # Parse input arguments
    training_data_path = os.path.expanduser(args.training_data)
    training_labels_path = os.path.expanduser(args.training_labels)
    testing_data_path = os.path.expanduser(args.testing_data)
    testing_labels_path = os.path.expanduser(args.testing_labels)
    attributes_path = os.path.expanduser(args.attributes)

    # Load data from relevant files
    # ***MODIFY CODE HERE***
    print(f"Loading training data from: {os.path.basename(training_data_path)}")
    xtrain = np.loadtxt(training_data_path, dtype=float, delimiter=",")
    print(f"Loading training labels from: {os.path.basename(training_labels_path)}")
    ytrain = np.loadtxt(training_labels_path, dtype=int)
    print(f"Loading testing data from: {os.path.basename(testing_data_path)}")
    xtest = np.loadtxt(testing_data_path, dtype=float,delimiter=",")
    print(f"Loading testing labels from: {os.path.basename(testing_labels_path)}")
    ytest = np.loadtxt(testing_labels_path, dtype=int)
    print(f"Loading attributes from: {os.path.basename(attributes_path)}")
    attributes = np.loadtxt(attributes_path, dtype=str)

    print("\n=======================")
    print("TRAINING")
    print("=======================")
    # Use a DecisionTreeClassifier to learn the full tree from training data
    print("Training the entire tree...")
    # ***MODIFY CODE HERE***
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(xtrain,ytrain)

    # Visualize the tree using matplotlib and plot_tree
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 5), dpi=150)
    # ***MODIFY CODE HERE***
    classNames = ["parkinson's","healthy"]
    plot_tree(clf,feature_names=attributes,class_names=classNames,filled=True,rounded=True)

    if args.save:
        filename = os.path.expanduser(os.path.join(THIS, 'tree.png'))
        print(f"  Saving to file: {os.path.basename(filename)}")
        plt.savefig(filename, bbox_inches='tight')
    plt.show(block=args.show)
    plt.close(fig)

    # Validating the root node of the tree by computing information gain
    print("Computing the information gain for the root node...")
    # ***MODIFY CODE HERE***
    index = clf.tree_.feature[0]  # index of the attribute that was determined to be the root node
    thold = clf.tree_.threshold[0]  # threshold on that attribute
    gain = information_gain(xtrain, ytrain, index, thold)
    print(f"  Root: {attributes[index]}<={thold:0.3f}, Gain: {gain:0.3f}")

    # Test the decision tree
    print("\n=======================")
    print("TESTING")
    print("=======================")
    # ***MODIFY CODE HERE***
    print("Predicting labels for training data...")
    ptrain = clf.predict(xtrain)
    print("Predicting labels for testing data...")
    ptest = clf.predict(xtest)

    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compare training and test accuracy
    # ***MODIFY CODE HERE***
    total_correct_train = 0
    for i in range(len(ptrain)):
        if ptrain[i] == ytrain[i]:
            total_correct_train += 1
    accuracy_train = total_correct_train / len(ptrain)

    total_correct_test = 0
    for i in range(len(ptest)):
        if ptest[i] == ytest[i]:
            total_correct_test += 1
    accuracy_test = total_correct_test / len(ptest)
    
    print(f"Training Accuracy: {total_correct_train}/{len(ptrain)} ({accuracy_train*100}%)")
    print(f"Testing Accuracy: {total_correct_test}/{len(ptest)} ({accuracy_test*100}%)")

    # Show the confusion matrix for test data
    # ***MODIFY CODE HERE***



    cm = confusion_matrix(ytest, ptest)
    print("Confusion matrix:")
    for i in cm:
        for j in i:
            print(f"{j:>4}", end="")
        print()

    # Debug (if requested)
    if args.debug:
        pdb.set_trace()

def information_gain(x, y, index, thold):
    """Compute the information gain on y for a continuous feature in x (using index) by applying a threshold (thold).

    NOTE: The threshold should be applied as 'less than or equal to' (<=)"""
    # ***MODIFY CODE HERE***

    e = entropy(y)
    ce = conditional_entropy(x, y, index, thold)
    gain = e - ce

    return gain

def entropy(y):
    """compute the entropy of y for the dataset"""
    count = 0
    for i in y:
        if i == 0:
            count += 1
    prob = count/len(y)
    e = -1 * prob * np.log2(prob) + -1 * (1-prob) * np.log2(1-prob)
    return e

def conditional_entropy(x, y, index, thold):
    """comput the conditional entropy of y given the attribute corresponding to the index"""
    counts = [0, 0, 0, 0, 0, 0]
    for i in range(len(x)):
        if x[i][index] <= thold:
            if y[i] == 0:
                counts[1] += 1
            else:
                counts[2] += 1
            counts[0] += 1
        else:
            if y[i] == 0:
                counts[4] += 1
            else:
                counts[5] += 1
            counts[3] += 1
    ce = (counts[0]/len(x))*(-1*(counts[1]/counts[0])*np.log2(counts[1]/counts[0]) + -1*(counts[2]/counts[0])*np.log2(counts[2]/counts[0])) + (counts[3]/len(x))*(-1*(counts[4]/counts[3])*np.log2(counts[4]/counts[3]) + -1*(counts[5]/counts[3])*np.log2(counts[5]/counts[3]))
    return ce

if __name__ == '__main__':
    main(parser.parse_args())
