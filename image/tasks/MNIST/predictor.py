import sys
sys.path.append("../../../utils")
import torch
import os
import pickle
import numpy as np
from MNISTSample import *
from models.TorchModels import *
from type_utils import *


CUDA = torch.cuda.is_available()
CATEGORY_NUM = 10


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
        model = torch.load(model_path)
        pass
    return model


def regularize_output(preds, category_num=10):
    ## convert scale outputs to vectors
    return preds
    new_preds = np.zeros((len(preds),category_num),dtype=np.int16) 
    for i,e in enumerate(preds):
        new_preds[i][e] = 1
    return new_preds




#================ ML models ====================

def do_test_sklearn_KNN(query_size):
    model_path = "models/mnist_knn_k15.pkl"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    test_X = test_X.view(-1,28*28) 
    testData = np.array(test_X)
    pred_y = model.predict_proba(testData)
    # pred_y = regularize_output(pred_y,CATEGORY_NUM)
    return pred_y


def do_test_sklearn_SVM(query_size):
    model_path = "models/mnist_svm_epoch20_new.pkl"
    model = load_model(model_path)
    test_X,_ = prepare_query(seed=0, size=query_size)
    test_X = test_X.view(-1,28*28) 
    testData = np.array(test_X)
    pred_y = model.predict_proba(testData)
    # pred_y = regularize_output(pred_y,CATEGORY_NUM)
    return pred_y

def do_test_pytorch_LR(query_size):
    model_path = "models/mnist_logistcReg_epoch5.pt"
    model = load_model(model_path)
    
    model.eval()
    with torch.no_grad():
        test_X,_ = prepare_query(seed=0, size=query_size)
        if CUDA:
            model = model.cuda()
            test_X = test_X.view(-1,28*28*1)
            test_X = test_X.cuda()
        pred_y,_ = model(test_X)    
        pred_y = torch.softmax(pred_y,dim=1)
    return pred_y.cpu().numpy()


#============ DL models =================
def do_test_pytorch_MLP(query_size):
    model_path = "models/mnist_mlp_epoch5.pt"
    model = load_model(model_path)
    
    model.eval()
    with torch.no_grad():
        test_X,_ = prepare_query(seed=0, size=query_size)
        if CUDA:
            model = model.cuda()
            test_X = test_X.view(-1,28*28*1)
            test_X = test_X.cuda()
        pred_y,_ = model(test_X)    
        pred_y = torch.softmax(pred_y,dim=1)
    return pred_y.cpu().numpy()


def do_test_pytorch_CNN(query_size):
    model_path = "models/mnist_cnn_epoch5.pt"
    model = load_model(model_path)
    
    model.eval()
    with torch.no_grad():
        test_X,_ = prepare_query(seed=0, size=query_size)
        if CUDA:
            model = model.cuda()
            test_X = test_X.cuda()
        pred_y,_ = model(test_X)    
        pred_y = torch.softmax(pred_y,dim=1)
    return pred_y.cpu().numpy()

def do_test_pytorch_RNN(query_size):
    model_path = "models/mnist_rnn_epoch5.pt"
    model = load_model(model_path)
    
    model.eval()
    with torch.no_grad():
        test_X,_ = prepare_query(seed=0, size=query_size)
        if CUDA:
            model = model.cuda()
            test_X = test_X.cuda().view(-1, 28, 28)    
            test_X = test_X.cuda()
        pred_y = model(test_X)    
        pred_y = torch.softmax(pred_y,dim=1)
    return pred_y.cpu().numpy()



        





def collect_outputs_of_type_label(q_size):

    save_path = "/home/user01/exps/PredictionMarket/outputs/MNIST/query_%d/type_label"%q_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # preds = do_test_sklearn_KNN(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # save_results_to_txt(preds, "%s/KNN.csv"%save_path)

    ## randomize SVM‘s outputs
    preds = do_test_sklearn_SVM(query_size=q_size)
    # preds = randomize_output(preds)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/SVM.csv"%save_path)

    # preds = do_test_pytorch_LR(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # save_results_to_txt(preds, "%s/LR.csv"%save_path)

    # ## randomize MLP‘s outputs
    preds = do_test_pytorch_MLP(query_size=q_size)
    # preds = randomize_output(preds)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/MLP.csv"%save_path)

    # preds = do_test_pytorch_CNN(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # save_results_to_txt(preds, "%s/CNN.csv"%save_path)

    # preds = do_test_pytorch_RNN(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # save_results_to_txt(preds, "%s/RNN.csv"%save_path)

def collect_outputs_of_type_rank(q_size):

    
    save_path = "/home/user01/exps/PredictionMarket/outputs/MNIST/query_%d/type_rank"%q_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # preds = do_test_sklearn_KNN(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/KNN.csv"%save_path)

    ## randomize SVM‘s outputs
    preds = do_test_sklearn_SVM(query_size=q_size)
    # preds = randomize_output(preds)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/SVM.csv"%save_path)

    # preds = do_test_pytorch_LR(query_size=q_size)
    # print(preds[0])
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/LR.csv"%save_path)

    # ## randomize MLP‘s outputs
    preds = do_test_pytorch_MLP(query_size=q_size)
    # preds = randomize_output(preds)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/MLP.csv"%save_path)

    # preds = do_test_pytorch_CNN(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/CNN.csv"%save_path)

    # preds = do_test_pytorch_RNN(query_size=q_size)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/RNN.csv"%save_path)


def collect_outputs_of_type_probs(q_size):

    save_path = "/home/user01/exps/PredictionMarket/outputs/MNIST/query_%d/type_probs"%q_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # preds = do_test_sklearn_KNN(query_size=q_size)
    # save_results_to_txt(preds, "%s/KNN.csv"%save_path)

    ## randomize SVM‘s outputs
    preds = do_test_sklearn_SVM(query_size=q_size)
    # preds = randomize_output(preds)
    save_results_to_txt(preds, "%s/SVM.csv"%save_path)

    # preds = do_test_pytorch_LR(query_size=q_size)
    # save_results_to_txt(preds, "%s/LR.csv"%save_path)

    # ## randomize MLP‘s outputs
    preds = do_test_pytorch_MLP(query_size=q_size)
    # preds = randomize_output(preds)
    save_results_to_txt(preds, "%s/MLP.csv"%save_path)

    # preds = do_test_pytorch_CNN(query_size=q_size)
    # save_results_to_txt(preds, "%s/CNN.csv"%save_path)

    # preds = do_test_pytorch_RNN(query_size=q_size)
    # save_results_to_txt(preds, "%s/RNN.csv"%save_path)



if __name__ == "__main__":
    q_sizes = [1000, 4000, 7000]
    
    # print("collecting %d querys..."%q_size)
    # print("collecting querys of type label...")
    # collect_outputs_of_type_label(q_size)
    for q_size in q_sizes:
        print("collecting querys of type label...")
        collect_outputs_of_type_label(q_size)
        print("collecting querys of type rank...")
        collect_outputs_of_type_rank(q_size)
        print("collecting querys of type probs...")
        collect_outputs_of_type_probs(q_size)
        print("collection done.")

    # preds = do_test_sklearn_KNN(query_size=q_size)
    # print(preds.argmax(axis=1))
    # nosie_preds = randomize_output(preds)
    # print(nosie_preds.argmax(axis=1))
    # print( nosie_preds == preds)

    # preds = do_test_sklearn_SVM(query_size=q_size)
    # print(preds)

    # preds = do_test_pytorch_LR(query_size=q_size)
    # print(preds.argmax(axis=1))

    # preds = do_test_pytorch_MLP(query_size=q_size)
    # print(preds.argmax(axis=1))

    # preds = do_test_pytorch_CNN(query_size=q_size)
    # print(preds.argmax(axis=1))
    # nosie_preds = randomize_output(preds)
    # print(nosie_preds.argmax(axis=1))
    # print( nosie_preds == preds)

    # preds = do_test_pytorch_RNN(query_size=q_size)
    # print(preds.argmax(axis=1))