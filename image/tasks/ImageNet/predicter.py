import sys
sys.path.append("../../../utils")
import json
import numpy as np
import torch 
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
import keras
import keras.applications as keras_application
from keras.preprocessing.image import ImageDataGenerator
import os
import utils
import warnings
from type_utils import *

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

CUDA = "cuda" if torch.cuda.is_available() else "cpu" 
BSIZE = 16


def load_data_torch(data_dir, batch_size=BSIZE):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BSIZE,
                                         shuffle=False, num_workers=2)

    return dataloader


def get_output_pytorch(model, dataloader):

    model.eval()
    outputs = None
    if CUDA:
        model = model.cuda()
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(dataloader):
            if CUDA:
                image = data.cuda()
                target = target.cuda()
            output = model(image)
            pred = output.cpu().data.numpy()
            # print(pred)
            if outputs is None:
                outputs = pred
            else:
                outputs = np.vstack((outputs, pred))
            # print(outputs.shape)
    # print(outputs.shape)
    # print(outputs[0])
    return outputs


def load_data_keras(dataset_path, preprocess_func, im_size = 224, batch_size=BSIZE):
    
    datagen = ImageDataGenerator(preprocessing_function= preprocess_func)
    test_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(im_size, im_size),
        color_mode="rgb",
        shuffle = False,
        class_mode='categorical',
        batch_size=batch_size,
    )

    return test_generator

    

def do_eval(model, data_dir):

   
    print("eval on %s"%data_dir)
    dataloader = load_data_torch(data_dir)
    
    model.eval()
    test_loss = 0
    test_correct = 0
    total = 0
    device = None
    use_cuda = True
    

    if use_cuda:
        device = torch.device('cuda')
        model = model.cuda()
    else:
        device = torch.device('cpu')
    
    criterion = nn.CrossEntropyLoss().to(device)
    evaluate(model, criterion, dataloader, "cuda" if use_cuda else "cpu", print_freq=100)


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg



#===== TEST model to get outputs - PyTorch ======
def do_test_pytorch_AlexNet(data_dir):
    dataloader = load_data_torch(data_dir)
    model = models.alexnet(pretrained=True)
    return get_output_pytorch(model, dataloader)
            

def do_test_pytorch_VGG16(data_dir):
    dataloader = load_data_torch(data_dir)
    model = models.vgg16(pretrained=True)
    return get_output_pytorch(model, dataloader)

def do_test_pytorch_VGG19(data_dir):
    dataloader = load_data_torch(data_dir)
    model = models.vgg19(pretrained=True)
    return get_output_pytorch(model, dataloader)

def do_test_pytorch_InceptionV3(data_dir):
    dataloader = load_data_torch(data_dir)
    model = models.inception_v3(pretrained=True)
    return get_output_pytorch(model, dataloader)

def do_test_pytorch_GoogLenet(data_dir):
    dataloader = load_data_torch(data_dir)
    model = models.googlenet(pretrained=True)
    return get_output_pytorch(model, dataloader)

def do_test_pytorch_ResNet50(data_dir):
    dataloader = load_data_torch(data_dir)
    model = models.resnet50(pretrained=True)
    return get_output_pytorch(model, dataloader)

def do_test_pytorch_ResNet101(data_dir):
    dataloader = load_data_torch(data_dir)
    model = models.resnet101(pretrained=True)
    return get_output_pytorch(model, dataloader)

def do_test_pytorch_MobileNetV2(data_dir):
    dataloader = load_data_torch(data_dir)
    model = models.mobilenet_v2(pretrained=True)
    return get_output_pytorch(model, dataloader)

def do_test_pytorch_DenseNet(data_dir):
    dataloader = load_data_torch(data_dir)
    model = models.densenet.densenet169(pretrained=True)
    return get_output_pytorch(model, dataloader)


#===== TEST model to get outputs - Keras ======
def do_test_keras_VGG16(data_dir):
    model = keras_application.vgg16.VGG16(weights="imagenet")
    preprocess_func = keras_application.vgg16.preprocess_input
    test_generator = load_data_keras(data_dir, preprocess_func)
    steps = int(data_dir.split("_")[-1]) / BSIZE
    preds = model.predict_generator(test_generator,verbose=1, steps=steps) 
    return preds

def do_test_keras_VGG19(data_dir):
    model = keras_application.vgg19.VGG19(weights="imagenet")
    preprocess_func = keras_application.vgg19.preprocess_input
    test_generator = load_data_keras(data_dir, preprocess_func)
    steps = int(data_dir.split("_")[-1]) / BSIZE
    preds = model.predict_generator(test_generator,verbose=1, steps=steps) 
    return preds

def do_test_keras_ResNet50(data_dir):
    model = keras_application.resnet50.ResNet50(weights="imagenet")
    preprocess_func = keras_application.resnet50.preprocess_input
    test_generator = load_data_keras(data_dir, preprocess_func)
    steps = int(data_dir.split("_")[-1]) / BSIZE
    preds = model.predict_generator(test_generator,verbose=1, steps=steps) 
    return preds

def do_test_keras_ResNet101(data_dir):
    model = keras_application.ResNet101(weights="imagenet")
    preprocess_func = keras_application.resnet_v2.preprocess_input
    test_generator = load_data_keras(data_dir, preprocess_func)
    steps = int(data_dir.split("_")[-1]) / BSIZE
    preds = model.predict_generator(test_generator,verbose=1, steps=steps) 
    return preds

def do_test_keras_InceptionV3(data_dir):
    model = keras_application.inception_v3.InceptionV3(weights="imagenet")
    preprocess_func = keras_application.inception_v3.preprocess_input
    test_generator = load_data_keras(data_dir, preprocess_func, im_size=299)
    steps = int(data_dir.split("_")[-1]) / BSIZE
    preds = model.predict_generator(test_generator,verbose=1, steps=steps) 
    return preds


def do_test_keras_DenseNet(data_dir):
    model = keras_application.densenet.DenseNet121(weights="imagenet")
    preprocess_func = keras_application.densenet.preprocess_input
    test_generator = load_data_keras(data_dir, preprocess_func)
    steps = int(data_dir.split("_")[-1]) / BSIZE
    preds = model.predict_generator(test_generator,verbose=1, steps=steps)    
    return preds


def do_test_keras_MobileNetV2(data_dir):
    model = keras_application.MobileNetV2(weights="imagenet")
    preprocess_func = keras_application.mobilenet.preprocess_input
    test_generator = load_data_keras(data_dir, preprocess_func)
    steps = int(data_dir.split("_")[-1]) / BSIZE
    preds = model.predict_generator(test_generator,verbose=1, steps=steps)    
    return preds


## ======================
def collect_outputs_of_type_label(dataset_path):

    q_size = int(dataset_path.split("_")[-1])

    save_path = "/home/user01/exps/PredictionMarket/outputs/ImageNet/query_%d/type_label"%q_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # add noise
    preds = do_test_pytorch_AlexNet(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    preds =  randomize_output(preds)
    save_results_to_txt(preds, "%s/pytorch_AlexNet.csv"%save_path)

    preds = do_test_pytorch_VGG16(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/pytorch_VGG16.csv"%save_path)

    ## add noise
    preds = do_test_pytorch_VGG19(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    preds =  randomize_output(preds)
    save_results_to_txt(preds, "%s/pytorch_VGG19.csv"%save_path)

    preds = do_test_pytorch_InceptionV3(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/pytorch_InceptionV3.csv"%save_path)

    preds = do_test_pytorch_GoogLenet(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/pytorch_GoogLenet.csv"%save_path)

    preds = do_test_pytorch_ResNet50(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/pytorch_ResNet50.csv"%save_path)

    preds = do_test_pytorch_ResNet101(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/pytorch_ResNet101.csv"%save_path)

    preds = do_test_pytorch_MobileNetV2(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/pytorch_MobileNetV2.csv"%save_path)

    preds = do_test_pytorch_DenseNet(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/pytorch_DenseNet.csv"%save_path)

    preds = do_test_keras_VGG16(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/keras_VGG16.csv"%save_path)

    preds = do_test_keras_VGG19(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/keras_VGG19.csv"%save_path)

    preds = do_test_keras_ResNet50(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/keras_ResNet50.csv"%save_path)

    preds = do_test_keras_InceptionV3(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/keras_InceptionV3.csv"%save_path)

    preds = do_test_keras_DenseNet(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    save_results_to_txt(preds, "%s/keras_DenseNet.csv"%save_path)

    preds = do_test_keras_MobileNetV2(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    preds =  randomize_output(preds)
    save_results_to_txt(preds, "%s/keras_MobileNetV2.csv"%save_path)


def collect_outputs_of_type_rank(dataset_path):

    q_size = int(dataset_path.split("_")[-1])

    save_path = "/home/user01/exps/PredictionMarket/outputs/ImageNet/query_%d/type_rank"%q_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ## add noise
    # preds = do_test_pytorch_AlexNet(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds =  randomize_output(preds)
    # save_results_to_txt(preds, "%s/pytorch_AlexNet.csv"%save_path)

    # preds = do_test_pytorch_VGG16(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/pytorch_VGG16.csv"%save_path)

    # ## add noise
    # preds = do_test_pytorch_VGG19(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds =  randomize_output(preds)
    # save_results_to_txt(preds, "%s/pytorch_VGG19.csv"%save_path)

    
    # preds = do_test_pytorch_InceptionV3(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/pytorch_InceptionV3.csv"%save_path)

    # preds = do_test_pytorch_GoogLenet(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/pytorch_GoogLenet.csv"%save_path)

    # preds = do_test_pytorch_ResNet50(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/pytorch_ResNet50.csv"%save_path)

    # preds = do_test_pytorch_ResNet101(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/pytorch_ResNet101.csv"%save_path)

    # preds = do_test_pytorch_MobileNetV2(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/pytorch_MobileNetV2.csv"%save_path)

    # preds = do_test_pytorch_DenseNet(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # save_results_to_txt(preds, "%s/pytorch_DenseNet.csv"%save_path)

    preds = do_test_keras_VGG16(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/keras_VGG16.csv"%save_path)

    preds = do_test_keras_VGG19(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/keras_VGG19.csv"%save_path)

    preds = do_test_keras_ResNet50(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/keras_ResNet50.csv"%save_path)

    preds = do_test_keras_InceptionV3(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/keras_InceptionV3.csv"%save_path)

    preds = do_test_keras_DenseNet(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    save_results_to_txt(preds, "%s/keras_DenseNet.csv"%save_path)

    ## add noise
    preds = do_test_keras_MobileNetV2(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    preds =  randomize_output(preds)
    save_results_to_txt(preds, "%s/keras_MobileNetV2.csv"%save_path)

def collect_outputs_of_type_probs(dataset_path):

    q_size = int(dataset_path.split("_")[-1])

    save_path = "/home/user01/exps/PredictionMarket/outputs/ImageNet/query_%d/type_probs_new"%q_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ## add noise
    # preds = do_test_pytorch_AlexNet(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # preds =  randomize_output(preds)
    # save_results_to_txt(preds, "%s/pytorch_AlexNet.csv"%save_path)

    # preds = do_test_pytorch_VGG16(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds, "%s/pytorch_VGG16.csv"%save_path)

    # ## add noise
    # preds = do_test_pytorch_VGG19(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # preds =  randomize_output(preds)
    # save_results_to_txt(preds, "%s/pytorch_VGG19.csv"%save_path)

    
    # preds = do_test_pytorch_InceptionV3(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds, "%s/pytorch_InceptionV3.csv"%save_path)

    # preds = do_test_pytorch_GoogLenet(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds, "%s/pytorch_GoogLenet.csv"%save_path)

    # preds = do_test_pytorch_ResNet50(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds, "%s/pytorch_ResNet50.csv"%save_path)

    # preds = do_test_pytorch_ResNet101(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds, "%s/pytorch_ResNet101.csv"%save_path)

    # preds = do_test_pytorch_MobileNetV2(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds, "%s/pytorch_MobileNetV2.csv"%save_path)

    # preds = do_test_pytorch_DenseNet(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds, "%s/pytorch_DenseNet.csv"%save_path)

    # preds = do_test_keras_VGG16(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds, "%s/keras_VGG16.csv"%save_path)

    # preds = do_test_keras_VGG19(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds, "%s/keras_VGG19.csv"%save_path)

    # preds = do_test_keras_ResNet50(dataset_path)
    # preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds, "%s/keras_ResNet50.csv"%save_path)

    preds = do_test_keras_InceptionV3(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/keras_InceptionV3.csv"%save_path)

    preds = do_test_keras_DenseNet(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds, "%s/keras_DenseNet.csv"%save_path)

    preds = do_test_keras_MobileNetV2(dataset_path)
    preds = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    preds =  randomize_output(preds)
    save_results_to_txt(preds, "%s/keras_MobileNetV2.csv"%save_path)


def collect_in_one(dataset_path):
        
    q_size = int(dataset_path.split("_")[-1])

    save_path_label = "/home/user01/exps/PredictionMarket/outputs/ImageNet/query_%d/type_label"%q_size
    save_path_rank = "/home/user01/exps/PredictionMarket/outputs/ImageNet/query_%d/type_rank"%q_size
    save_path_probs = "/home/user01/exps/PredictionMarket/outputs/ImageNet/query_%d/type_probs"%q_size
    if not os.path.exists(save_path_label):
        os.makedirs(save_path)
    if not os.path.exists(save_path_rank):
        os.makedirs(save_path)
    if not os.path.exists(save_path_probs):
        os.makedirs(save_path)

    ## add noise
    preds = do_test_pytorch_AlexNet(dataset_path)
    # preds =  randomize_output(preds)
    preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds_label, "%s/pytorch_AlexNet.csv"%save_path_label)
    save_results_to_txt(preds_rank, "%s/pytorch_AlexNet.csv"%save_path_rank)
    save_results_to_txt(preds_probs, "%s/pytorch_AlexNet.csv"%save_path_probs)

    # preds = do_test_pytorch_VGG16(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/pytorch_VGG16.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/pytorch_VGG16.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/pytorch_VGG16.csv"%save_path_probs)

    ## add noise
    preds = do_test_pytorch_VGG19(dataset_path)
    # preds =  randomize_output(preds)
    preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds_label, "%s/pytorch_VGG19.csv"%save_path_label)
    save_results_to_txt(preds_rank, "%s/pytorch_VGG19.csv"%save_path_rank)
    save_results_to_txt(preds_probs, "%s/pytorch_VGG19.csv"%save_path_probs)

    
    # preds = do_test_pytorch_InceptionV3(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/pytorch_InceptionV3.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/pytorch_InceptionV3.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/pytorch_InceptionV3.csv"%save_path_probs)

    # preds = do_test_pytorch_GoogLenet(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/pytorch_GoogLenet.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/pytorch_GoogLenet.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/pytorch_GoogLenet.csv"%save_path_probs)

    # preds = do_test_pytorch_ResNet50(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/pytorch_ResNet50.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/pytorch_ResNet50.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/pytorch_ResNet50.csv"%save_path_probs)

    # preds = do_test_pytorch_ResNet101(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/pytorch_ResNet101.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/pytorch_ResNet101.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/pytorch_ResNet101.csv"%save_path_probs)

    # preds = do_test_pytorch_MobileNetV2(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/pytorch_MobileNetV2.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/pytorch_MobileNetV2.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/pytorch_MobileNetV2.csv"%save_path_probs)

    # preds = do_test_pytorch_DenseNet(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/pytorch_DenseNet.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/pytorch_DenseNet.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/pytorch_DenseNet.csv"%save_path_probs)

    # preds = do_test_keras_VGG16(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/keras_VGG16.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/keras_VGG16.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/keras_VGG16.csv"%save_path_probs)

    # preds = do_test_keras_VGG19(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/keras_VGG19.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/keras_VGG19.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/keras_VGG19.csv"%save_path_probs)

    # preds = do_test_keras_ResNet50(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/keras_ResNet50.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/keras_ResNet50.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/keras_ResNet50.csv"%save_path_probs)

    # preds = do_test_keras_InceptionV3(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/keras_InceptionV3.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/keras_InceptionV3.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/keras_InceptionV3.csv"%save_path_probs)

    # preds = do_test_keras_DenseNet(dataset_path)
    # preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    # preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    # preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    # save_results_to_txt(preds_label, "%s/keras_DenseNet.csv"%save_path_label)
    # save_results_to_txt(preds_rank, "%s/keras_DenseNet.csv"%save_path_rank)
    # save_results_to_txt(preds_probs, "%s/keras_DenseNet.csv"%save_path_probs)

    ## add noise 
    preds = do_test_keras_MobileNetV2(dataset_path)
    # preds =  randomize_output(preds)
    preds_label = convert_output_type(preds, OUTPUT_TYPE.TYPE_LABEL)
    preds_rank = convert_output_type(preds, OUTPUT_TYPE.TYPE_RANK)
    preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
    save_results_to_txt(preds_label, "%s/keras_MobileNetV2.csv"%save_path_label)
    save_results_to_txt(preds_rank, "%s/keras_MobileNetV2.csv"%save_path_rank)
    save_results_to_txt(preds_probs, "%s/keras_MobileNetV2.csv"%save_path_probs)


def save_ground_truth():

    q_size = 5000
    for q_size in [5000,10000,20000]:
        test_generator = load_data_keras("../../dataset/imagenet/sample/query_%d"%q_size,None)   
        y = test_generator.classes
        np.savetxt("ImageNet_%d.csv"%q_size, y, fmt="%d", delimiter=",")





if __name__ == "__main__":
    pass

    # =============== Test PyTorch models ==============
    # preds = do_test_pytorch_AlexNet(dataset_path)
    # print(preds.argmax(axis=1)[:10])

    # preds = do_test_pytorch_VGG16(dataset_path)
    # print(preds.argmax(axis=1)[:10])

    # preds = do_test_pytorch_VGG19(dataset_path)
    # print(preds.argmax(axis=1)[:10])

    # preds = do_test_pytorch_InceptionV3(dataset_path)
    # print(preds.argmax(axis=1)[:10])

    # preds = do_test_pytorch_GoogLenet(dataset_path)
    # print(preds.argmax(axis=1)[:10])

    # preds = do_test_pytorch_ResNet50(dataset_path)
    # print(preds.argmax(axis=1)[:10])    

    # preds = do_test_pytorch_ResNet101(dataset_path)
    # print(preds.argmax(axis=1)[:10])  

    # preds = do_test_pytorch_MobileNetV2(dataset_path)
    # print(preds.argmax(axis=1)[:10])  

    # preds = do_test_pytorch_DenseNet(dataset_path)
    # print(preds.argmax(axis=1)[:10])  


    # =============== Test Keras models ==============
    # preds = do_test_keras_VGG16(dataset_path)
    # print(preds.argmax(axis=1)[:10])  

    # preds = do_test_keras_VGG19(dataset_path)
    # print(preds.argmax(axis=1)[:10])  

    # preds = do_test_keras_ResNet50(dataset_path)
    # print(preds.argmax(axis=1)[:10])  

    # Keras's version is too old to support a in-build ResNet101
    # preds = do_test_keras_ResNet101(dataset_path)
    # print(preds.argmax(axis=1)[:10])  

    # preds = do_test_keras_InceptionV3(dataset_path)
    # print(preds.argmax(axis=1)[:10])  

    # preds = do_test_keras_DenseNet(dataset_path)
    # print(preds.argmax(axis=1)[:10])  

    # preds = do_test_keras_MobileNetV2(dataset_path)
    # print(preds.argmax(axis=1)[:10])  
    

    dataset_paths = [
        # "../../dataset/imagenet/sample/query_5000",
        # "../../dataset/imagenet/sample/query_10000",
        "../../dataset/imagenet/sample/query_20000",
    ]
        
    for dataset_path in dataset_paths:
        collect_in_one(dataset_path)

