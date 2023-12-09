from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from DecisionTreeLearning import DecisionTreeNode, DecisionTreeClassifier
import pickle
import argparse
from sklearn.model_selection import train_test_split
import re


class AdaBooster:

    def __init__(self, iterations):
        self.iterations = iterations
        self.classifiers = []
        self.errors = []
        self.weights_alpha = []

    def fit(self, data, labels):
        samples_count = data.shape[0]
        weights = np.ones(samples_count) / samples_count

        for iter in range(self.iterations):
            sample_indices = np.random.choice(samples_count, size=samples_count, replace=True, p=weights)
            sampled_data, sampled_labels = data[sample_indices], labels[sample_indices]

            classifier = DecisionTreeClassifier(max_depth=1)
            classifier.train_classifier(sampled_data, sampled_labels)

            predictions = classifier.predict(data)
            error = np.sum(weights * (predictions != labels)) / np.sum(weights)
            alpha = np.log((1 - error) / (error + 1e-10))

            self.weights_alpha.append(alpha)
            weights *= np.exp(alpha * (predictions != labels))
            weights /= np.sum(weights)
            self.classifiers.append(classifier)

            if iter % 100 == 0:
                print(f"Iteration {iter}: Error = {error}")

    def predict(self, data):
        df_predictions = pd.DataFrame(index=range(len(data)), columns=range(self.iterations))
        for i, classifier in enumerate(self.classifiers):
            predictions = classifier.predict(data) * self.weights_alpha[i]
            df_predictions[i] = predictions
        final_pred = (1 * np.sign(df_predictions.T.sum())).astype(int)
        return final_pred

    def __calc_error(self, true_labels, predicted_labels, weights):
        return (sum(weights * (np.not_equal(true_labels, predicted_labels)).astype(int))) / sum(weights)

    def __update_weights(self, alpha, weights, true_labels, predicted_labels):
        return weights * np.exp(alpha * (np.not_equal(true_labels, predicted_labels)).astype(int))

    def __compute_alpha(self, error):
        return np.log((1 - error) / error)



def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

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



def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def predict(model_file, input_file):
    model = load_model(model_file)
    data_matrix = create_feature_matrix(input_file, action="predict")
    X = data_matrix
    predictions = model.predict(X)
    return predictions

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

    if args.command == "train" and args.learning_type == "ada":
        data_matrix = create_feature_matrix(args.examples, action="train")
        X = data_matrix[:, :-1]
        y = data_matrix[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=123)

        adaboost = AdaBooster(100)  # Set the number of iterations as needed
        adaboost.fit(X_train, y_train)  # Pass DecisionTree class to fit method
        y_pred_test = adaboost.predict(X_test)
        test_acc = accuracy(y_test, y_pred_test)
        print("Test Accuracy: ", test_acc)


    elif args.command == "predict":
        predictions = predict(args.model_file, args.input_file)
        with open("predictions2.txt", 'w') as file:
            for pred in predictions:
                file.write("en\n" if pred == 1 else "nl\n")

if __name__ == "__main__":
    main()