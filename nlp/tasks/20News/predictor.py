import sys
sys.path.append("../../../utils")
import random
import os
import pickle
import keras
import numpy as np
from NewsSampler import *
from sklearn.feature_extraction.text import TfidfVectorizer
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from type_utils import *

import warnings


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

CATEGORY_NUM = 20

def data_preprocess_TFIDF(X):
    ## TODO: need test
    ### TFIDF: for DNN - keras
    vectorizer_x = pickle.load(open("/home/user01/exps/PredictionMarket/nlp/tasks/20News/models/TFIDFVectorizer.pkl","rb"))
    X_processed = vectorizer_x.transform(X).toarray()
    print(X_processed.shape)
    return X_processed

def data_preprocess_Tokenizer(X, MAX_NB_WORDS=75000, MAX_SEQUENCE_LENGTH=500):
    ## TODO: need test
    ## Glove
    text = X.copy()
    # tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    # tokenizer.fit_on_texts(text)
    tokenizer = pickle.load(open("/home/user01/exps/PredictionMarket/nlp/tasks/20News/models/Tokenizer.pkl","rb"))
    sequences = tokenizer.texts_to_sequences(text)
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    indices = np.arange(text.shape[0])
    text = text[indices]
    # print(text.shape)
    X_processed = text
    return X_processed

def load_model(model_path):
    model = None
    _, model_suffix = os.path.splitext(model_path)
    print("loading model %s"%model_path)
    if model_suffix in ['.h5']:
        print("load keras models")
        model = keras.models.load_model(model_path)
    elif model_suffix in ['.pkl']:
        with open(model_path,"rb") as f:
            model = pickle.load(f)
    elif model_suffix in ['.pt']:
        #model = torch.load(model_path)
        pass
    return model

def regularize_output(preds, category_num=10):
    ## convert scale outputs to vectors
    new_preds = np.zeros((len(preds),category_num),dtype=np.int16) 
    for i,e in enumerate(preds):
        new_preds[i][e] = 1
    return new_preds


#=================== ML models ====================

def do_test_sklearn_knn(query_size):
    model_path = "models/20News_KNN.pkl"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    pred_y = model.predict_proba(test_X)    
    # pred_y = regularize_output(pred_y,CATEGORY_NUM)
    return pred_y


def do_test_sklearn_svm(query_size):
    model_path = "models/20News_SVM_probs.pkl"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    pred_y = model.predict_proba(test_X)    
    # pred_y = regularize_output(pred_y,CATEGORY_NUM)
    return pred_y


def do_test_sklearn_bagging(query_size):
    model_path = "models/20News_bagging.pkl"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    pred_y = model.predict_proba(test_X)    
    # pred_y = regularize_output(pred_y,CATEGORY_NUM)
    return pred_y

def do_test_sklearn_boost(query_size):
    model_path = "models/20News_boost.pkl"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    pred_y = model.predict_proba(test_X)    
    # pred_y = regularize_output(pred_y,CATEGORY_NUM)
    return pred_y


def do_test_sklearn_random_forest(query_size):
    model_path = "models/20News_randomForest.pkl"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    pred_y = model.predict_proba(test_X)    
    # pred_y = regularize_output(pred_y,CATEGORY_NUM)
    return pred_y

def do_test_sklearn_decision_tree(query_size):
    model_path = "models/20News_DecisionTree.pkl"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    pred_y = model.predict_proba(test_X)    
    # pred_y = regularize_output(pred_y,CATEGORY_NUM)
    return pred_y





#================= DL models =====================

def do_test_keras_DNN(query_size):
    model_path = "models/20News_DNN.h5"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    test_X = data_preprocess_TFIDF(test_X)
    pred_y = model.predict(test_X)
    return pred_y

def do_test_keras_CNN(query_size):
    model_path = "models/20News_CNN.h5"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    test_X = data_preprocess_Tokenizer(test_X)
    pred_y = model.predict(test_X)    
    return pred_y

def do_test_keras_RNN(query_size):
    model_path = "models/20News_RNN.h5"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    test_X = data_preprocess_Tokenizer(test_X)
    pred_y = model.predict(test_X)    
    return pred_y

def do_test_keras_RCNN(query_size):
    model_path = "models/20News_RCNN.h5"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    test_X = data_preprocess_Tokenizer(test_X)
    pred_y = model.predict(test_X)    
    return pred_y


#================ collecting ===============
def collect_outputs_of_type_label(q_size):
    save_path = "/home/user01/exps/PredictionMarket/outputs/20News/query_%d/type_label"%q_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    preds = do_test_sklearn_svm(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/svm.csv"%save_path)

    preds = do_test_sklearn_knn(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/knn.csv"%save_path)

    preds = do_test_sklearn_bagging(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/bagging.csv"%save_path)

    preds = do_test_sklearn_boost(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/boost.csv"%save_path)

    preds = do_test_sklearn_random_forest(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/srandom_forest.csv"%save_path)

    preds = do_test_sklearn_decision_tree(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/decision_tree.csv"%save_path)

    preds = do_test_keras_DNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/DNN.csv"%save_path)

    preds = do_test_keras_CNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/CNN.csv"%save_path)

    preds = do_test_keras_RNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    preds =  randomize_output(preds)
    save_results_to_txt(preds, "%s/RNN.csv"%save_path)

    preds = do_test_keras_RCNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/RCNN.csv"%save_path)




def collect_outputs_of_type_rank(q_size):


    save_path = "/home/user01/exps/PredictionMarket/outputs/20News/query_%d/type_rank"%q_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    preds = do_test_sklearn_svm(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/svm.csv"%save_path)

    preds = do_test_sklearn_knn(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/knn.csv"%save_path)

    preds = do_test_sklearn_bagging(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/bagging.csv"%save_path)

    preds = do_test_sklearn_boost(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/boost.csv"%save_path)

    preds = do_test_sklearn_random_forest(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/srandom_forest.csv"%save_path)

    preds = do_test_sklearn_decision_tree(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/decision_tree.csv"%save_path)


    preds = do_test_keras_DNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/DNN.csv"%save_path)

    preds = do_test_keras_CNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/CNN.csv"%save_path)

    preds = do_test_keras_RNN(query_size=q_size)
    preds =  randomize_output(preds)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/RNN.csv"%save_path)

    preds = do_test_keras_RCNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/RCNN.csv"%save_path)




def collect_outputs_of_type_probs(q_size):


    save_path = "/home/user01/exps/PredictionMarket/outputs/20News/query_%d/type_probs"%q_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    preds = do_test_sklearn_svm(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/svm.csv"%save_path)

    preds = do_test_sklearn_knn(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/knn.csv"%save_path)

    preds = do_test_sklearn_bagging(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/bagging.csv"%save_path)

    preds = do_test_sklearn_boost(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/boost.csv"%save_path)

    preds = do_test_sklearn_random_forest(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/srandom_forest.csv"%save_path)

    preds = do_test_sklearn_decision_tree(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/decision_tree.csv"%save_path)

    preds = do_test_keras_DNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/DNN.csv"%save_path)

    preds = do_test_keras_CNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/CNN.csv"%save_path)

    preds = do_test_keras_RNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    preds =  randomize_output(preds)
    save_results_to_txt(preds, "%s/RNN.csv"%save_path)

    preds = do_test_keras_RCNN(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/RCNN.csv"%save_path)





def collect_in_ones(q_size):
    save_path_label = "/home/user01/exps/PredictionMarket/outputs/20News/query_%d/type_label"%q_size
    save_path_rank = "/home/user01/exps/PredictionMarket/outputs/20News/query_%d/type_rank"%q_size
    save_path_probs = "/home/user01/exps/PredictionMarket/outputs/20News/query_%d/type_probs"%q_size
    if not os.path.exists(save_path_label):
        os.makedirs(save_path_label)
    if not os.path.exists(save_path_rank):
        os.makedirs(save_path_rank)
    if not os.path.exists(save_path_probs):
        os.makedirs(save_path_probs)
        
    preds = do_test_sklearn_svm(query_size=q_size)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/svm.csv"%save_path_label)

    # preds = do_test_sklearn_knn(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # save_results_to_txt(preds, "%s/knn.csv"%save_path_label)

    # preds = do_test_sklearn_bagging(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # save_results_to_txt(preds, "%s/bagging.csv"%save_path_label)

    # preds = do_test_sklearn_boost(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # save_results_to_txt(preds, "%s/boost.csv"%save_path_label)

    # preds = do_test_sklearn_random_forest(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # save_results_to_txt(preds, "%s/srandom_forest.csv"%save_path_label)

    # preds = do_test_sklearn_decision_tree(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # save_results_to_txt(preds, "%s/decision_tree.csv"%save_path_label)

    # # ======= DL models ======

    # preds = do_test_keras_DNN(query_size=q_size)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank  = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/DNN.csv"%save_path_label)
    # save_results_to_txt(preds_rank , "%s/DNN.csv"%save_path_rank )
    # save_results_to_txt(preds_probs, "%s/DNN.csv"%save_path_probs)

    # preds = do_test_keras_CNN(query_size=q_size)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/CNN.csv"%save_path_label)
    # save_results_to_txt(preds_rank , "%s/CNN.csv"%save_path_rank )
    # save_results_to_txt(preds_probs, "%s/CNN.csv"%save_path_probs)

    preds = do_test_keras_RNN(query_size=q_size)
    preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # preds =  randomize_output(preds)
    save_results_to_txt(preds_label, "%s/RNN.csv"%save_path_label)
    save_results_to_txt(preds_rank , "%s/RNN.csv"%save_path_rank )
    save_results_to_txt(preds_probs, "%s/RNN.csv"%save_path_probs)

    # preds = do_test_keras_RCNN(query_size=q_size)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/RCNN.csv"%save_path_label)
    # save_results_to_txt(preds_rank , "%s/RCNN.csv"%save_path_rank )
    # save_results_to_txt(preds_probs, "%s/RCNN.csv"%save_path_probs)



if __name__ == "__main__":

    # q_size = 10

    q_sizes = [1000, 3000, 5000]
    # # q_sizes = [5000]

    for q_size in q_sizes:
        collect_in_ones(q_size)

    # preds = do_test_sklearn_svm(query_size=q_size)
    # print(preds)

    # preds = do_test_sklearn_knn(query_size=q_size)
    # print(preds.argmax(axis=1))

    # preds = do_test_sklearn_bagging(query_size=q_size)
    # print(preds.argmax(axis=1))

    # preds = do_test_sklearn_boost(query_size=q_size)
    # print(preds)

    # preds = do_test_sklearn_random_forest(query_size=q_size)
    # print(preds)

    # preds = do_test_sklearn_decision_tree(query_size=q_size)
    # print(preds)


    #============= DL models ================
    # preds = do_test_keras_DNN(query_size=q_size)
    # print(preds.argmax(axis=1))

    # preds = do_test_keras_CNN(query_size=q_size)
    # print(preds.argmax(axis=1))

    # preds = do_test_keras_RNN(query_size=q_size)
    # print(preds.argmax(axis=1))

    # preds = do_test_keras_RCNN(query_size=q_size)
    # print(preds.argmax(axis=1))


