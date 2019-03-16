from numpy import dot, matmul
from scipy.stats import logistic

'''

This is simple gradient descent algorithm with 1-layer 

The output of the class gives 0 - 1 probability area for potential buy trade

License: for education purpose only.

'''

class GradientDescent():

    def __init__(self, number_of_inputs, indicators, best_trades, no_iter = 10000):
        '''
        Initialize the object
        
        :param number_of_inputs: number of columns from data used to predict the trade deal
        :param indicators: actual data columns
        :param best_trades: y_hat columns
        :param no_iter: by default = 10000
        '''

        self.number_of_inputs = number_of_inputs
        self.gradient_weights = [[0.05]] * number_of_inputs
        self.train_gd_algorithm(indicators, best_trades, no_iter)


    def sigmoid_function(self, x):
        '''
        
        :param x: array of data to perform activation function on
        :return: sigmoid prob value
        
        '''

        sig_value = logistic.cdf(x)

        return sig_value

    def sigmoid_function_adjustment(self, x):
        '''
        sigmoid function as follow:

        f(x) = 1/(1-exp(1)^(-x))

        (d/dx)f(x) = ( exp(1)^x * (1 + exp(1)^x) - exp(1)^x * exp(1)^x  ) / (1 + exp(1)^2  ) =
                   = exp(1) / (1 + exp(1)^x)^2 =
                   = f(x)*(1-f(x))

        '''
        adjustment = x * (1 - x)
        
        return adjustment

    def train_gd_algorithm(self, indicators, best_trades, number_of_training_iterations):
        '''
        This is a function to train the gd model by predicting and adjusting the model's weights

        :param indicators: data column
        :param best_trades: y_hat column
        :param number_of_training_iterations:
        :return:
        '''

        for epo in range(number_of_training_iterations):
            print("Current epoch is :", str(epo))

            current_model_results = self.forward(indicators)
            current_error_values = best_trades - current_model_results
            adjustment = matmul(indicators.T, current_error_values * self.sigmoid_function_adjustment(current_model_results))
            self.gradient_weights += adjustment

	t = helper_method(X,X,X)
	
	return #restricted content
	
    def forward(self, inputs):
        '''
        Perform sigmoid activation forward prediction on data

        :param inputs: current data
        :return: weights to adjust
        '''

        dot_product = dot(inputs, self.gradient_weights)
        a = self.sigmoid_function(dot_product)
        return a

    def sample_income(X,X,X):
	#restricted content
	return #restricted content

    def helper_method(X,X,X):
	#restricted content

if __name__ == "__main__":

    pass
    #restricted content
