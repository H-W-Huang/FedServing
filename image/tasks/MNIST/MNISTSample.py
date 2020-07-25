import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random
import numpy as np

def prepare_query(size, seed=0):

    random.seed(seed)
    indice = random.sample(range(10000), size)

    test_data = dsets.MNIST(root='/home/user01/exps/PredictionMarket/image/dataset/MNIST', train=False, transform=transforms.ToTensor())
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[indice]
    test_x = test_x/255
    test_y = test_data.test_labels.numpy()[indice]

    # reshaping depends on specific models
    # test_x = test_x.view(-1,28*28)
    # print(test_x.shape)
    # print("sample labels (first 10):\n%s"%str(test_y[:10]))
    return test_x, test_y


def test():
    prepare_query(5)


def save_ground_truth(q_size):
    _,y = prepare_query(q_size)

    np.savetxt("MNIST_%d.csv"%q_size, y.astype(np.int16), fmt="%d", delimiter=",")


if __name__ == "__main__":
    save_ground_truth(1000)
    save_ground_truth(4000)
    save_ground_truth(7000)