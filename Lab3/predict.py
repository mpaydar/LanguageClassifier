# import pickle
# import DecisionTreeLearning
# import numpy as np
# import re
# import argparse
# from sklearn.model_selection import train_test_split
#
#
#
# def text_preprocessing(sentence):
#     english_pronouns=["he","she","it","they","i","you"]
#     english_determinatives=["the","a","an","this","those","that","these","his","her","some","any","enough","each","every","either","neither","which","whose","what"]
#     english_coordinators=["for","and","nor","but","or","yet","so"]
#     modal_verbs=["can","could","shall","should","will","would","may","might","must"]
#     english_auxiliaries=["is", "am", "are", "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did"]
#
#     words = set(re.findall(r'\b\w+\b', sentence.lower()))
#     features = {
#         "hasEnglishPronouns": 1 if words.intersection(english_pronouns) else -1,
#         "hasEnglishDeterminative": 1 if words.intersection(english_determinatives) else -1,
#         "hasEnglishAuxiliary": 1 if words.intersection(english_auxiliaries) else -1,
#         "hasEnglishCoordinators": 1 if words.intersection(english_coordinators) else -1,
#         "hasPreposition": 1 if words.intersection(modal_verbs) else -1
#     }
#     return features
#
#
#
#
#
# def create_feature_matrix(examples,action):
#
#
#
#     if action=="train":
#         text_matrix = []
#         label_matrix = []
#         with open(examples, 'r', encoding='utf-8') as file:
#             sentences = file.read().split('\n')
#             for sentence in sentences:
#                 preprocessed_data=text_preprocessing(sentence)
#                 text_matrix.append(list(preprocessed_data.values()))
#             feature_matrix=np.column_stack((text_matrix,label_matrix))
#     else:
#         with open(examples, 'r', encoding='utf-8') as file:
#             sentences = file.read().split('\n')
#             text_matrix=[]
#             for sentence in sentences:
#                 if sentence:
#                     preprocessed_data=text_preprocessing(sentence)
#                     text_matrix.append(list(preprocessed_data.values()))
#             feature_matrix=np.array(text_matrix)
#     return feature_matrix
#
#
#
#
#
#
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Machine Learning Model Training and Prediction")
#
#     # Train parser
#     parser.add_argument("data_action", help="Are you training or testing?")
#     parser.add_argument("examples", type=str, help="File containing labeled examples")
#     parser.add_argument("hypothesis", type=str, help="File name to write your model to")
#     parser.add_argument("learning_type", type=str, choices=['dt', 'ada'],
#                         help="Type of learning algorithm (dt or ada)")
#
#     args = parser.parse_args()
#
#     if args.data_action == "predict":
#             data_matrix = create_feature_matrix(args.examples)
#             clf=pickle.load(args.hypothesis)
#             y_pred1 = clf.predict(data_matrix)
#
#
#
# if __name__ == "__main__":
#     main()
