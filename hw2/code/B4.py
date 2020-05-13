import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn

to_tensor = transforms.ToTensor()

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=to_tensor)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=to_tensor)
train_loader = torch.utils.data.DataLoader(mnist_trainset,
                                          batch_size=128,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_testset,
                                          batch_size=128,
                                          shuffle=True)


def train_CrossEntropy(l, num_epochs, train_loader):
    
    # initialize W
    W = torch.zeros(784, 10, requires_grad = True)
    # define loss function
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # iterate through batches
        for inputs, labels in tqdm(iter(train_loader)):
            # flatten images
            inputs = torch.flatten(inputs, start_dim=1, end_dim=3)
            # compute predictions
            preds = torch.matmul(inputs, W)
            # compute loss
            loss = torch.nn.functional.cross_entropy(preds, labels)
            # computes derivatives of the loss with respect to W
            loss.backward()
            # gradient descent update
            W.data = W.data - l * W.grad
            W.grad.zero_()
        print("Loss\t{}".format(loss))
    return W

def train_MSE(l, num_epochs, train_loader):
    
    # initialize W
    W = 0.001*torch.rand(784, 10, requires_grad = True)
    for epoch in range(num_epochs):
        # iterate through batches
        for inputs, labels in tqdm(iter(train_loader)):
            # flatten images
            inputs = torch.flatten(inputs, start_dim=1, end_dim=3)
            # compute predictions
            preds = torch.matmul(inputs, W)
            # convert labels to one-hot labels
            y_onehot = torch.zeros(preds.shape[0], preds.shape[1])
            y_onehot.scatter_(1, labels.unsqueeze(1), 1)
            # compute loss
            loss = torch.mean(torch.norm((preds-y_onehot), dim = 1)**2)/2
            # computes derivatives of the loss with respect to W
            grad = torch.autograd.grad(loss, [W])
            # gradient descent update
            W.data = W.data - l * grad[0]
        print("Loss\t{}".format(loss))
    return W
        


def compute_accuracy(W, data_loader):
    acc = 0
    for inputs, labels in tqdm(iter(test_loader)):
        inputs = torch.flatten(inputs, start_dim=1, end_dim=3)
        preds = torch.argmax(torch.matmul(inputs, W),1)
        acc += torch.sum(preds == labels)
        
    acc = acc.to(dtype=torch.float)/len(test_loader.dataset)
    return(acc)


if __name__ == '__main__':


    W_CE = train_CrossEntropy(0.001, 50, train_loader)
    W_MSE = train_MSE(0.001, 50, train_loader)

    acc_CE_test = compute_test_accuracy(W_CE, test_loader)
    acc_MSE_test = compute_test_accuracy(W_MSE, test_loader)
    acc_CE_train = compute_test_accuracy(W_CE, train_loader)
    acc_MSE_train = compute_test_accuracy(W_MSE, train_loader)
    print('CE test accuracy: ', acc_CE_test)
    print('MSE test accuracy: ', acc_MSE_test)
    print('CE train accuracy: ', acc_CE_train)
    print('MSE train accuracy: ', acc_MSE_train)