import random
from sklearn.datasets import fetch_20newsgroups
import numpy as np


def prepare_query(size=5000,seed=0):
    
    newsgroups_test = fetch_20newsgroups(subset='test',data_home="/home/user01/exps/PredictionMarket/nlp/dataset/20News")
    assert size < len(newsgroups_test.data) and size != 0 and size is not None
    
    random.seed(0)
    random_indice = random.sample(range(len(newsgroups_test.data)), size)
    
    test_X = [newsgroups_test.data[i] for i in random_indice]
    test_y = newsgroups_test.target[random_indice] 
    print("sample labels (first 10):\n%s"%str(test_y[:10]))
    return test_X, test_y

def save_ground_truth(q_size):
    _,y = prepare_query(q_size)

    np.savetxt("20News_%d.csv"%q_size, y, fmt="%d", delimiter=",")


if __name__ == "__main__":

    save_ground_truth(1000)
    save_ground_truth(3000)
    save_ground_truth(5000)