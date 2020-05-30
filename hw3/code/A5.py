####################################
#HW3, Problem A5
####################################
import numpy as np
import scipy
import matplotlib.pyplot as plt 
import seaborn as sns
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn


to_tensor = transforms.ToTensor()

mnist_trainset = datasets.MNIST(
	root='./data', train=True, download=True, transform=to_tensor)
mnist_testset = datasets.MNIST(
	root='./data', train=False, download=True, transform=to_tensor)
train_loader = torch.utils.data.DataLoader(
	mnist_trainset,batch_size=128,shuffle=True)
test_loader = torch.utils.data.DataLoader(
	mnist_testset,batch_size=128, shuffle=True)


def compute_accuracy(test_loader, net_type, W0=None, W1=None, W2=None, b0=None, b1=None, b2=None):
	acc = 0
	loss = 0
	for inputs, labels in tqdm(iter(test_loader)):
		inputs = torch.flatten(inputs, start_dim=1, end_dim=3)
		if net_type == 'wide':
			logits = wide_network(inputs, W0, W1, b0, b1)
			preds = torch.argmax(logits,1)
		elif net_type == 'deep':
			logits = deep_network(inputs, W0, W1, W2, b0, b1, b2)
			preds = torch.argmax(logits,1)

		acc += torch.sum(preds == labels)
		loss += torch.nn.functional.cross_entropy(logits, labels, size_average = False)
		
	loss = loss/len(test_loader.dataset)
	acc = acc.to(dtype=torch.float)/len(test_loader.dataset)
	return acc, loss



def wide_network(input, W0, W1, b0, b1):
	layer1 = torch.matmul(input, W0.T)+b0
	layer2 = torch.matmul(nn.functional.relu(layer1), W1.T) +b1
	return layer2



def deep_network(input, W0, W1, W2, b0, b1, b2):
	layer1 = torch.matmul(input, W0.T)+b0
	layer2 = torch.matmul(nn.functional.relu(layer1), W1.T) +b1
	layer3 = torch.matmul(nn.functional.relu(layer2), W2.T) +b2
	return layer3


def train_wide_network(data_loader, l, num_epochs, n_neurons = 64, input_dim = 784):
	# initialize parameters randomly
	alpha = 1/np.sqrt(input_dim)
	W0 = -2*alpha* torch.rand(n_neurons, input_dim) + alpha
	W0.requires_grad = True
	W1 = -2*alpha* torch.rand(10, n_neurons) + alpha
	W1.requires_grad = True
	b0 = -2*alpha* torch.rand(n_neurons) + alpha
	b0.requires_grad = True
	b1 = -2*alpha* torch.rand(10) + alpha
	b1.requires_grad = True
	# define loss function
	params = [W0, W1, b0, b1]
	optimizer = torch.optim.Adam(params, lr=l)
	loss_array = []
	for epoch in range(num_epochs):
		acc = 0
		loss_array.append(0)
		# iterate through batches
		for inputs, labels in tqdm(iter(data_loader)):
			# flatten images
			inputs = torch.flatten(inputs, start_dim=1, end_dim=3)
			# compute predictions
			logits = wide_network(inputs, W0, W1, b0, b1)
			preds = torch.argmax(logits,1)
			acc += torch.sum(preds == labels)
			# compute loss
			loss = torch.nn.functional.cross_entropy(logits, labels, size_average = False)
			# computes derivatives of the loss with respect to W
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_array[epoch] += loss
		loss_array[epoch] = loss_array[epoch]/len(data_loader.dataset)
		print("For epoch %s training loss is %s" % (epoch, loss_array[epoch]))
		acc = acc.to(dtype=torch.float)/len(data_loader.dataset)
		print("For epoch %s training accuracy is %s" % (epoch, acc))
		if acc>0.99:
			return loss_array, W0, W1, b0, b1
	return loss_array, W0, W1, b0, b1



def train_deep_network(data_loader, l, num_epochs, n_neurons = 32, input_dim = 784):
	# initialize parameters randomly
	alpha = 1/np.sqrt(input_dim)
	W0 = -2*alpha* torch.rand(n_neurons, input_dim) + alpha
	W0.requires_grad = True
	W1 = -2*alpha* torch.rand(n_neurons, n_neurons) + alpha
	W1.requires_grad = True
	W2 = -2*alpha* torch.rand(10, n_neurons) + alpha
	W2.requires_grad = True
	b0 = -2*alpha* torch.rand(n_neurons) + alpha
	b0.requires_grad = True
	b1 = -2*alpha* torch.rand(n_neurons) + alpha
	b1.requires_grad = True
	b2 = -2*alpha* torch.rand(10) + alpha
	b2.requires_grad = True
	# define loss function
	params = [W0, W1, W2, b0, b1, b2]
	optimizer = torch.optim.Adam(params, lr=l)
	loss_array = []
	for epoch in range(num_epochs):
		loss_array.append(0)
		acc = 0
		# iterate through batches
		for inputs, labels in tqdm(iter(data_loader)):
			# flatten images
			inputs = torch.flatten(inputs, start_dim=1, end_dim=3)
			# compute predictions
			logits = deep_network(inputs, W0, W1, W2, b0, b1, b2)
			preds = torch.argmax(logits,1)
			acc += torch.sum(preds == labels)
			# compute loss
			loss = torch.nn.functional.cross_entropy(logits, labels, size_average = False)
			# computes derivatives of the loss with respect to W
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_array[epoch] += loss
		loss_array[epoch] = loss_array[epoch]/len(data_loader.dataset)
		print("For epoch %s training loss is %s" % (epoch, loss_array[epoch]))
		acc = acc.to(dtype=torch.float)/len(data_loader.dataset)
		print("For epoch %s training accuracy is %s" % (epoch, acc))
		if acc>0.99:
			return loss_array, W0, W1, W2, b0, b1, b2
	return loss_array, W0, W1, W2, b0, b1, b2



if __name__ == '__main__':

	#Wide net:
	loss_array_wide, W0, W1, b0, b1 = train_wide_network(train_loader, 0.001, 500, 
		n_neurons = 64, input_dim = 784)
	test_acc, test_loss = compute_accuracy(test_loader, net_type ='wide', 
		W0=W0, W1=W1, W2=None, b0=b0, b1=b1, b2=None)
	print('Test accuracy for wide network is', test_acc)
	print('Test loss for wide network is', test_loss)

	number_parameters_wide = np.prod(W0.shape) + np.prod(
		W1.shape) + np.prod(b0.shape) + np.prod(b1.shape)
	print('Number of parameters for wide network', number_parameters_wide)

	#Deep net:
	loss_array_deep, W0, W1, W2, b0, b1, b2 = train_deep_network(
		train_loader, 0.001, 500, n_neurons = 32, input_dim = 784)
	test_acc, test_loss = compute_accuracy(
		test_loader, net_type ='deep', W0=W0, W1=W1, W2=W2, b0=b0, b1=b1, b2=b2)
	print('Test accuracy for deep network is', test_acc)
	print('Test loss for deep network is', test_loss)

	number_parameters_deep = np.prod(W0.shape) + np.prod(W1.shape) + np.prod(
		W2.shape) + np.prod(b0.shape) + np.prod(b1.shape) + np.prod(b2.shape)
	print('Number of parameters for deep network', number_parameters_deep)


	#Plot the results
	x_plot_wide = range(len(loss_array_wide))
	x_plot_deep = range(len(loss_array_deep))
	plt.figure(figsize = (15,10))
	plt.plot(x_plot_wide, loss_array_wide, '-o', label = 'wide network')
	plt.plot(x_plot_deep, loss_array_deep, '-o', label = 'deep network')
	plt.title('Training Loss for wide and deep networks')
	plt.legend()
	plt.xlabel('epoch')
	plt.ylabel('error')
	plt.savefig('figures/A5_training_plots.pdf')
	plt.show()

