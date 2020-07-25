import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
# import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 5   # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False
MNIST_PATH = "../../../dataset/MNIST/"

CUDA = True


# Mnist digits dataset
if not(os.path.exists(MNIST_PATH)) or not os.listdir(MNIST_PATH):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root=MNIST_PATH,
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# plot one example


train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root=MNIST_PATH, train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:10000].cuda()


print(train_data.train_data.size())                 # (60000, 28, 28)
print(test_data.test_data.size())                 # (10000, 28, 28)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


# cnn = CNN()
cnn = torch.load("mnist_cnn_epoch5.pt")
print(cnn)  # net architecture

# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
# loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# if CUDA:
#     cnn = cnn.cuda()
# print(cnn)
# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
#         if CUDA:
#             b_x = x.cuda()
#             b_y = y.cuda()
#         output = cnn(b_x)[0]               # cnn output
#         loss = loss_func(output, b_y)   # cross entropy loss
#         optimizer.zero_grad()           # clear gradients for this training step
#         loss.backward()                 # backpropagation, compute gradients
#         optimizer.step()                # apply gradients

#         if step % 50 == 0:
#             test_output, last_layer = cnn(test_x)
#             pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
#             accuracy = float((pred_y == test_y.cpu().data.numpy()).astype(int).sum()) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.5f' % accuracy)


print("saving model...")
# torch.save(cnn,"mnist_cnn_epoch5.pt")
print("done!")




