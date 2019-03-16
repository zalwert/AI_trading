import torch
import torch.nn as nn
torch.manual_seed(3)

"""

This is a 3-layer deep network class based on the PyTorch library

Number of hidden neurons to be set in initialization of the class

"""


class Net(nn.Module):

    def __init__(self, D_in, H1, H2, D_out):
        """
        Initialization of the class

        :param D_in: input parameters from the data
        :param H1: number of hidden neurons
        :param H2: number of hidden neurons
        :param D_out: outputs (as 0 - 1)
        """

        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    def forward(self, x):
        '''

        Activation function on each layer is based on the sigmoid function

        Please refer to: https://en.wikipedia.org/wiki/Sigmoid_function

        '''
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)

        return x

def train(model, criterion, train_loader, test_loader, optimizer, epochs = 10000):
    '''

    This is a basic train function in PyTorch library

    :param model: Deep Network model with desired number of inputs, hidden neurons and outputs
    :param criterion: nn.CrossEntropyLoss() - as a probability
    :param train_loader: data to train
    :param test_loader: data to test
    :param optimizer: optimizer class for the model
    :param epochs: number of epochs (by default 1000)
    :return: data_model and income based on correctness of the model

    '''

    data_model = {'training_loss': [], 'test_income': []}

    for epo in range(epochs):
        print("The currenct epoch is: ", epo)

        for i, (x_data, y) in enumerate(train_loader):

            optimizer.zero_grad()
            z = model(x_data)
            model_loss = criterion(z, y)
            model_loss.backward()
            optimizer.step()
            data_model['training_loss'].append(model_loss.data.item())

            print("The loss for a given epoch is: ", model_loss.data.item())

        correct = 0
        for i, (x_data, y) in enumerate(test_loader):
            print("test starts...")

            yhat = model(x_data)
            temp_o, lab = torch.max(yhat, 1)
            correct += (lab == y).sum().item()
            income = (correct / len(test_loader))
            data_model['test_income'].append(income)

            print("Based on test sample the income is: ", income)

	    income_check(X,X,X)

    return data_model, income

    def income_check(X,X,X):
	#restricted content


if __name__ == "__main__":

    pass
    #restricted content
