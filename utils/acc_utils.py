import numpy as np
import os

def cal_acc(results_file, truth_file):
    """
        only calculate the top-1 accuracy.
    """
    
    truth = np.genfromtxt(truth_file,delimiter=",").astype(np.int16)
    preds = np.genfromtxt(results_file,delimiter=",")    

    correct = ( preds.argmax(axis=1) == truth ).sum()
    total = len(truth)


    return correct/total


def cal_acc_imagenet_rank(results_file, truth_file, k):
    """
        only calculate the top-1 accuracy.
    """
    
    truth = np.genfromtxt(truth_file,delimiter=",").astype(np.int16)
    preds = np.genfromtxt(results_file,delimiter=",")    

    top_10s =  preds.argsort(axis=1)
    print(top_10s.shape)
    correct = 0
    for i,t in enumerate(truth):
        correct += 1 if t in top_10s[i][-k:][::-1] else 0
    # correct = ( preds.argmax(axis=1) == truth ).sum()
    total = len(truth)
    return correct/total
    return 0

if __name__ == "__main__":

    Ts = {
        "MNIST": {
            "q_sizes" : [1000, 4000, 7000],
            # "q_sizes" : [7000],
            "noise_levels" : [0,3,5],
            "iter_nums": [20]
            # "iter_nums": [1,2,3,4,5,6,7,8,9,10]
        },

        # "20News": {
        #     "q_sizes" : [1000,3000,5000],
        #     "noise_levels" : [0,5,9],
        #     "iter_nums": [20]
        #     # "iter_nums": [1,2,3,4,5,6,7,8,9,10,12,14,16]
        # },

        # "ImageNet": {
        #     "q_sizes" : [5000,10000,16000],
        #     "noise_levels" : [0, 7, 14],
        #     "iter_nums": [20]
        #     # "iter_nums": [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]
        # }
    }
    otypes = ["label", "rank", "probs"]
    # otypes = ["label"]

    for t,v in Ts.items():
        print("============= %s ============="%(t))
        for noise_level in v["noise_levels"]:
            print("> noise level is : %d"%noise_level)
            for iter_num in v["iter_nums"]:
                # print("> iter nums is : %d"%iter_num)
                for otype in otypes:
                    for q_size in v["q_sizes"]:
                        # print("current q_size: %d"%q_size)
                        results_file = "../outputs_combined/combination_results_new/%s/noise_%d/%s_%d_%s_iter_%d.csv"%(t,noise_level,t,q_size,otype,iter_num)
                        truth_file = "../outputs/%s/ground_truth/%s_%d.csv"%(t,t,q_size)
                        acc = cal_acc(results_file,truth_file)
                        # acc = cal_acc_imagenet_rank(resutls_file,truth_file,k)
                        print("> %s\t\t%s - acc:\t%f"%(os.path.split(results_file)[-1],otype,acc))
