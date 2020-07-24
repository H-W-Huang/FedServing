import sys
sys.path.append('../utils')
import numpy as np
from type_utils import *


def make_noise(category_num,size, otype=OUTPUT_TYPE.TYPE_LABEL):
    ran = np.random.rand(size,category_num)
    preds = softmax_2d(ran)
    if otype == OUTPUT_TYPE.TYPE_LABEL:
        preds = convert_to_type_label(preds)
    elif otype == OUTPUT_TYPE.TYPE_RANK:
        preds = convert_to_type_rank(preds)
    return preds
    

def produce(category_num, q_size, task, model, output_type):
    save_path = "/home/user01/exps/PredictionMarket/outputs/%s/query_%d/type_%s/%s_noise.csv"%(task,q_size,output_type, model)
    otype = None
    if output_type == "label":
        otype = OUTPUT_TYPE.TYPE_LABEL
    elif output_type == "rank":
        otype = OUTPUT_TYPE.TYPE_RANK
    elif output_type == "probs":
        otype = OUTPUT_TYPE.TYPE_PROBS
    assert otype is not None
    preds = make_noise(category_num, q_size, otype)
    print("saving to %s..."%(save_path))
    save_results_to_txt(preds,save_path)

if __name__ == "__main__":

    MNIST_models = {
        "KNN",
        "SVM",
        "CNN",
        "RNN",
        "LR",
        "MLP"
    }
    MNIST_qsizes = [1000,4000,7000]
    MNIST_category_num = 10

    t20News_models = {
        "knn",            
        "svm",            
        "decision_tree",  
        "srandom_forest", 
        "bagging",        
        "boost",          
        "DNN",            
        "CNN",            
        "RNN" ,           
        "RCNN"    
    }
    t20News_qsizes = [1000,3000,5000]
    t20News_category_num = 20


    ImageNet_models = {
        "pytorch_AlexNet",
        "pytorch_DenseNet",
        "pytorch_GoogLenet",
        "pytorch_InceptionV3",
        "pytorch_MobileNetV2",
        "pytorch_ResNet101",
        "pytorch_ResNet50",
        "pytorch_VGG16",
        "pytorch_VGG19",
        "keras_DenseNet",
        "keras_InceptionV3",
        "keras_MobileNetV2",
        "keras_ResNet50",
        "keras_VGG16",
        "keras_VGG19"
    }
    ImageNet_qsizes = [5000,10000,20000]
    ImageNet_category_num = 1000


    output_types = ["label", "rank", "probs"]
    # ## MNIST
    # for output_type in output_types:
    #     for q_size in MNIST_qsizes:
    #         for model in MNIST_models:
    #             produce(MNIST_category_num, q_size, "MNIST", model , output_type)

    ## 20News
    # for output_type in output_types:
    #     for q_size in t20News_qsizes:
    #         for model in t20News_models:
    #             produce(t20News_category_num, q_size, "20News", model , output_type)

    ## ImageNet
    for output_type in output_types:
        for q_size in ImageNet_qsizes:
            for model in ImageNet_models:
                produce(ImageNet_category_num, q_size, "ImageNet", model , output_type) 




        
        
        
        
        
        
        
        
        
        
















