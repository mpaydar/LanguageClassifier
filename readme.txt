#  Polyphonic AI: Multilingual Music Selector 

This is my personal ongoing project and I am in the progress of making the web application that is going to use the data model to dynamically understand the specified language user is typing and then use the prediction in creative way to show the user list of top musics in that specified predicted language. The user can select one of the musics for playback. 


For Decision Tree Learning:
python DecisionTreeLearning.py train training.txt model.pkl dt

For prediction:
python DecisionTreeLearning.py predict model.pkl test.txt


For AdaBoost Training:
python adaboost.py train training.txt model.pkl ada

For Prediction:
python adaboost.py predict model.pkl verification.txt











