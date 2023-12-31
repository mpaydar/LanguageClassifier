import numpy as np
import pandas as pd
from collections import Counter
import re
import pickle
import argparse
from sklearn.model_selection import train_test_split



def calculate_entropy(y):
    hist = np.bincount(y)
    probabilities = hist / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

class DecisionTreeNode:
    def __init__(self, split_criteria=None, threshold_value=None, left_branch=None, right_branch=None, leaf_value=None):
        self.split_criteria = split_criteria
        self.threshold_value = threshold_value
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.leaf_value = leaf_value

    def is_terminal_node(self):
        return self.leaf_value is not None

class DecisionTreeClassifier:
    def __init__(self, min_split_size=2, max_depth=100, feature_limit=None):
        self.min_split_size = min_split_size
        self.max_depth = max_depth  # Ensure max_depth is set up correctly
        self.feature_limit = feature_limit
        self.tree_base = None

    def train_classifier(self, features, targets):
        self.feature_limit = min(self.feature_limit, features.shape[1]) if self.feature_limit else features.shape[1]
        self.tree_base = self.build_tree(features, targets, 0)  # Add a depth parameter starting from 0

    def build_tree(self, features, targets, current_depth=0):

        # Check if the current depth exceeds the max depth limit
        if current_depth >= self.max_depth or len(set(targets)) == 1 or len(targets) < self.min_split_size:
            if len(targets) == 0:
                # Handle empty target list
                return DecisionTreeNode(leaf_value=None)
            most_common_label = Counter(targets).most_common(1)[0][0]
            return DecisionTreeNode(leaf_value=most_common_label)

        selected_features = np.random.choice(features.shape[1], self.feature_limit, replace=False)
        best_feature, best_threshold = self.determine_best_split(features, targets, selected_features)
        left_split, right_split = self.split_data(features[:, best_feature], best_threshold)
        left_child = self.build_tree(features[left_split], targets[left_split], current_depth + 1)
        right_child = self.build_tree(features[right_split], targets[right_split], current_depth + 1)
        return DecisionTreeNode(best_feature, best_threshold, left_child, right_child)

    def determine_best_split(self, features, targets, feature_indices):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feature_idx in feature_indices:
            thresholds = np.unique(features[:, feature_idx])
            for threshold in thresholds:
                gain = self.information_gain(targets, features[:, feature_idx], threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def information_gain(self, targets, feature_column, threshold):
        parent_entropy = calculate_entropy(targets)
        left_idxs, right_idxs = self.split_data(feature_column, threshold)
        n = len(targets)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = calculate_entropy(targets[left_idxs]), calculate_entropy(targets[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entropy - child_entropy
        return ig

    def split_data(self, feature_column, threshold):
        left_idxs = np.where(feature_column <= threshold)[0]
        right_idxs = np.where(feature_column > threshold)[0]
        return left_idxs, right_idxs

    def predict(self, features):
        return np.array([self.traverse_tree(sample, self.tree_base) for sample in features])

    def traverse_tree(self, sample, node):
        if node.is_terminal_node():
            return node.leaf_value
        if sample[node.split_criteria] <= node.threshold_value:
            return self.traverse_tree(sample, node.left_branch)
        return self.traverse_tree(sample, node.right_branch)

def measure_accuracy(actual, predicted):
    return np.sum(actual == predicted) / len(actual)

def reading_file(text_file):
    sentences = []
    with open(text_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        sentences = [line.rstrip() for line in lines]
    return np.array(sentences)

def text_preprocessing(sentence):
    english_pronouns=["he","she","it","they","i","you"]
    english_determinatives=["the","a","an","this","those","that","these","his","her","some","any","enough","each","every","either","neither","which","whose","what"]
    english_coordinators=["for","and","nor","but","or","yet","so"]
    modal_verbs=["can","could","shall","should","will","would","may","might","must"]
    english_auxiliaries=["is", "am", "are", "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did"]

    words = set(re.findall(r'\b\w+\b', sentence.lower()))
    features = {
        "hasEnglishPronouns": 1 if words.intersection(english_pronouns) else -1,
        "hasEnglishDeterminative": 1 if words.intersection(english_determinatives) else -1,
        "hasEnglishAuxiliary": 1 if words.intersection(english_auxiliaries) else -1,
        "hasEnglishCoordinators": 1 if words.intersection(english_coordinators) else -1,
        "hasPreposition": 1 if words.intersection(modal_verbs) else -1
    }
    return features


english_rules = [
    # Add your English rules here
    ["the", "and", "in", "is", "of", "to", "it", "that", "with", "as"],
    ["and", "but", "or", "if", "while", "although", "because", "since", "unless"],
    ["in", "on", "at", "by", "with", "about", "for", "of", "from", "to"],
    ["he",'she','it','what','which','that'],
    ["be", "have", "do", "say", "get", "make", "go", "know", "take", "see","come", "think", "look", "want", "give", "use", "find", "tell", "ask","work", "seem", "feel", "try", "leave", "call"]
]



def label_econding(label):
    label_encoding=[]
    if label=='en':
        return 1
    else:
        return 0


def create_feature_matrix(examples, action):
    if action == "train":
        text_matrix = []
        label_matrix = []
        with open(examples, 'r', encoding='utf-8') as file:
            sentences = file.read().split('\n')
            for sentence in sentences:
                if sentence:
                    lang, text = sentence.split('|')
                    preprocessed_data = text_preprocessing(text)
                    encoded_label = label_econding(lang)
                    label_matrix.append(encoded_label)
                    text_matrix.append(list(preprocessed_data.values()))
            feature_matrix = np.column_stack((text_matrix, label_matrix))
    else:  # For prediction
        text_matrix = []
        with open(examples, 'r', encoding='utf-8') as file:
            sentences = file.read().split('\n')
            for sentence in sentences:
                if sentence:
                    preprocessed_data = text_preprocessing(sentence)
                    text_matrix.append(list(preprocessed_data.values()))
            feature_matrix = np.array(text_matrix)
    return feature_matrix


def predict(model_file, input_file):
    # Load model
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    # Create feature matrix for prediction
    X_test = create_feature_matrix(input_file, action="predict")

    # Perform prediction
    predictions = model.predict(X_test)
    return predictions


def save_model(model,file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)









def main():
    parser = argparse.ArgumentParser(description="Machine Learning Model Training and Prediction")
    subparsers = parser.add_subparsers(dest='command')

    # Train parser
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument("examples", type=str, help="File containing labeled examples")
    train_parser.add_argument("hypothesis_out", type=str, help="File name to write your model to")
    train_parser.add_argument("learning_type", type=str, choices=['dt', 'ada'], help="Type of learning algorithm (dt or ada)")

    # Predict parser
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument("model_file", type=str, help="Trained model file")
    predict_parser.add_argument("input_file", type=str, help="Input file for prediction")

    args = parser.parse_args()

    if args.command == "train" and args.learning_type == "dt":
        data_matrix = create_feature_matrix(args.examples, action="train")
        features = data_matrix[:, :-1]
        labels = data_matrix[:, -1]
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=123)
        classifier = DecisionTreeClassifier()
        classifier.train_classifier(features_train, labels_train)
        accuracy = measure_accuracy(labels_train, classifier.predict(features_train))
        # print("Training Accuracy: ", accuracy)
        save_model(classifier, args.hypothesis_out)

    elif args.command == "predict":
        predictions = predict(args.model_file, args.input_file)
        with open("predictions.txt", 'w') as file:
            for prediction in predictions:
                file.write("en\n" if prediction == 1 else "nl\n")

if __name__ == "__main__":
    main()
