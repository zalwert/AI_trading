import torch
import torch.nn as nn
from AI_Gradient_Descent import GradientDescent
from numpy import array
from Download_data import Get_data
from Dataset_torch import Dataset_torch
from Deep_Network_Torch import Net, train
from spark_statistics import spark_stats
from rbm import test_rbm

'''

This is the main class of the program:

Part1. one layer gradient descent algorithm
Part2. three layer deep learning network in pytorch
Part3. restricted boltzmana machine (code not shared)

License. for education purposes only.

'''

if __name__ == "__main__":

    #Download data
    sciezka_dane = "/Users/XXX/DAT_XLSX_EURUSD_M1_201901.xlsx"
    Get_data = Get_data(sciezka_dane, 25, 5, mins = [10,20,30,60,120])
    x_train, x_test, y_train, y_test, data, data_torch, ALL = Get_data.get_train_test(ts=0.33, rs=42)

    #Spark statistics helper
    data_stats = spark_stats(sciezka=sciezka_dane, csv=False)
    print(data_stats.summ)

    #Part1. one layer gradient descent algorithm
    gd = GradientDescent(number_of_inputs=len(x_train.columns),
                         indicators=array(x_train.values.tolist()),
                         best_trades=array([y_train.values.tolist()]).T,
                         no_iter=10000)

    print("Gradient descent results: ")
    print(gd.gradient_weights)
    buy_list = []
    income = 0
    for x in range(len(x_test)):
        buy_list.append(gd.forward(x_test.values.tolist()[x]))

    maxi = max(buy_list)[0]
    mini = min(buy_list)[0]
    for x in buy_list:
        if x > (maxi - ((maxi - mini)/100*50)):
            print(x)
            print(buy_list.index(x))
            index = buy_list.index(x)
            income += ALL.OPEN_Bid.iloc[index+40] - ALL.OPEN_Bid.iloc[index]
    print("Income from part 1 equals = ", income)
    print("##############")

    #Part2. three layer deep learning network in pytorch
    print("Deep network using pytorch")

    split = int(len(data_torch)/6)*4
    train_torch = data_torch.iloc[:split]
    test_torch = data_torch.iloc[split:]

    train_dataset = Dataset_torch(train_torch, transform=True)
    test_dataset = Dataset_torch(test_torch, transform=True)

    """
    Cross Entropy Loss combines nn.LogSoftmax() and NLLLoss()
    
    """
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=82, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=24, shuffle=False)

    input_dim, hidden_dim1, hidden_dim2, output_dim = 36, 36 * 2, 36, 10

    model = Net(input_dim, hidden_dim1, hidden_dim2, output_dim)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    training_results = train(model=model, criterion=criterion, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, epochs=10)

    #Part3. Restricted Boltzman Machine
    result_model = test_rbm()

    income_rbm = 0
    for index in range(9900):
        print(index)
        res = result_model[index][:37].mean()
        print(res)

        if res > 0.55:

            income_rbm += ALL.OPEN_Bid.iloc[index + 10] - ALL.OPEN_Bid.iloc[index]


    print("Income is:", income_rbm)




