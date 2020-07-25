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
MNIST_PATH = "../../dataset/MNIST/"

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



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(28*28,28*28),
            nn.Linear(28*28,10)
        )

    def forward(self, x):
        output = self.mlp(x)
        return output, x    # return x for visualization

mlp = MLP()
print(mlp)  # net architecture

optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

if CUDA:
    mlp = mlp.cuda()
print(mlp)
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        if CUDA:
            b_x = x.view(-1,28*28).cuda()
            b_y = y.cuda()
        output = mlp(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output, last_layer = mlp(test_x.view(-1,28*28))
            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
            accuracy = float((pred_y == test_y.cpu().data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.5f' % accuracy)


print("saving model...")
torch.save(mlp,"mnist_mlp_epoch5.pt")
print("done!")




