import os
import numpy as np
from enum import Enum, unique

@unique
class OUTPUT_TYPE(Enum):
    TYPE_LABEL = 0
    TYPE_RANK = 1
    TYPE_PROBS = 2

def softmax_2d(x):
    mx = np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(x - mx)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator/denominator

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

def randomize_output(preds):
    otype = determind_output_type(preds)
    print("Got output type: %s"%str(otype))

    nosie =  np.random.rand(*preds.shape)  
    print("adding nosie")
    # preds = softmax(preds+nosie) ## effect seems slight
    preds = softmax_2d(nosie) ## effect seems slight

    if otype == OUTPUT_TYPE.TYPE_LABEL:
        preds = convert_to_type_label(preds)
    elif otype == OUTPUT_TYPE.TYPE_RANK:
        preds = convert_to_type_rank(preds)
    return preds


def determind_output_type(preds):
    otype = None

    assert len(preds.shape)==2 and len(preds) !=0 
    
    sample = preds[0]
    # print(sample)
    # print(sorted(sample))
    # print(sample.sum())
    # print(np.sum(range(0,len(sample))))
    # if sample.sum() == np.sum(range(0,len(sample))):
    if sample.sum() == np.sum(range(1,len(sample)+1)) or sample.sum() == np.sum(range(0,len(sample))):
        otype = OUTPUT_TYPE.TYPE_RANK
    else:
        if "int" in str(type(sample[0])):
            otype = OUTPUT_TYPE.TYPE_LABEL
        elif "float" in str(type(sample[0])):
            otype = OUTPUT_TYPE.TYPE_PROBS
    return otype


def convert_to_type_label(preds):
    new_preds = np.zeros_like(preds,dtype=np.int16) 
    for i,e in enumerate(preds):
        new_preds[i][np.argmax(e)] = 1
    return new_preds

def convert_to_type_rank(preds):
    new_preds = []
    for i in range(len(preds)):
        preds_argsort = np.argsort(preds[i])
        p = np.zeros_like(preds[i],dtype=np.int16)
        for j,e in enumerate(preds_argsort):
            p[e] = j+1
        new_preds.append(p)
    return np.array(new_preds)


def convert_output_type(preds, target_type):
        
    new_preds = None
    assert  target_type in OUTPUT_TYPE, "Unsupport output type!" 
    
    src_otype = determind_output_type(preds)
    print("src output type is %s"%str(src_otype))


    ## type 1 - information is insufficient to do the conversion
    if src_otype == OUTPUT_TYPE.TYPE_LABEL:
        print("information is insufficient to do the conversion")
        # if target_type === OUTPUT_TYPE.TYPE_RANK:
        #     pass
        # elif target_type === OUTPUT_TYPE.TYPE_PROBS:
        #     pass
        new_preds = preds
    ## type 2 
    elif src_otype == OUTPUT_TYPE.TYPE_RANK:
        if target_type == OUTPUT_TYPE.TYPE_LABEL:
            new_preds = convert_to_type_label(preds)
        elif target_type == OUTPUT_TYPE.TYPE_RANK:
            new_preds = preds
        elif target_type == OUTPUT_TYPE.TYPE_PROBS:
            print("information is insufficient to do the conversion")
            new_preds = preds 
    ## type 3 
    elif src_otype == OUTPUT_TYPE.TYPE_PROBS:
        if target_type == OUTPUT_TYPE.TYPE_LABEL:
            new_preds = convert_to_type_label(preds)
        elif target_type == OUTPUT_TYPE.TYPE_RANK:
            # print("converting ranks")
            # print(preds)
            new_preds = convert_to_type_rank(preds)
            # print(new_preds)
        elif target_type == OUTPUT_TYPE.TYPE_PROBS:
            new_preds = softmax_2d(preds)

    return new_preds

def save_results_to_txt(preds,save_path):
    otype = determind_output_type(preds)
    output_fmt = "%.9f" if otype == OUTPUT_TYPE.TYPE_PROBS else "%d"
    with open(save_path,'w') as f:
        np.savetxt(f,preds, fmt=output_fmt, delimiter=",")
    print("Results saved to %s"%save_path)

