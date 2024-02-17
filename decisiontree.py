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
    xtrain = file_read(training_data_path)
    print(f"Loading training labels from: {os.path.basename(training_labels_path)}")
    ytrain = file_read(training_labels_path)
    print(f"Loading testing data from: {os.path.basename(testing_data_path)}")
    xtest = file_read(testing_data_path)
    print(f"Loading testing labels from: {os.path.basename(testing_labels_path)}")
    ytest = file_read(testing_labels_path)
    print(f"Loading attributes from: {os.path.basename(attributes_path)}")
    attributes = [-1]

    print("\n=======================")
    print("TRAINING")
    print("=======================")
    # Use a DecisionTreeClassifier to learn the full tree from training data
    print("Training the entire tree...")
    # ***MODIFY CODE HERE***
    clf = -1

    # Visualize the tree using matplotlib and plot_tree
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 5), dpi=150)
    # ***MODIFY CODE HERE***
    # plot_tree(clf)

    if args.save:
        filename = os.path.expanduser(os.path.join(THIS, 'tree.png'))
        print(f"  Saving to file: {os.path.basename(filename)}")
        plt.savefig(filename, bbox_inches='tight')
    plt.show(block=args.show)
    plt.close(fig)

    # Validating the root node of the tree by computing information gain
    print("Computing the information gain for the root node...")
    # ***MODIFY CODE HERE***
    index, thold = -1, -1
    # index = clf.tree_.feature[0]  # index of the attribute that was determined to be the root node
    # thold = clf.tree_.threshold[0]  # threshold on that attribute
    gain = information_gain(xtrain, ytrain, index, thold)
    print(f"  Root: {attributes[index]}<={thold:0.3f}, Gain: {gain:0.3f}")

    # Test the decision tree
    print("\n=======================")
    print("TESTING")
    print("=======================")
    # ***MODIFY CODE HERE***
    print("Predicting labels for training data...")
    ptrain = -1
    print("Predicting labels for testing data...")
    ptest = -1

    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compare training and test accuracy
    # ***MODIFY CODE HERE***
    accuracy_train = -1
    accuracy_test = -1
    print(f"Training Accuracy: ?/? (?%)")
    print(f"Testing Accuracy: ?/? (?%)")

    # Show the confusion matrix for test data
    # ***MODIFY CODE HERE***
    print("Confusion matrix:")

    # Debug (if requested)
    if args.debug:
        pdb.set_trace()

def information_gain(x, y, index, thold):
    """Compute the information gain on y for a continuous feature in x (using index) by applying a threshold (thold).

    NOTE: The threshold should be applied as 'less than or equal to' (<=)"""
    
    # ***MODIFY CODE HERE***
    gain = -1

    return gain

def file_read(file):
    """Reads the file, and outputs an array, with each row being a line from the file"""
    arr = []
    with open(file, "r") as f:
        for line in f:
            arr.append(line.rstrip("\n").split(","))
    f.close()
    return arr

if __name__ == '__main__':
    main(parser.parse_args())
