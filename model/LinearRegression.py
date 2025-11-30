import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class  LinearRegression(object):
    def __init__(self, learning_rate, epochs, weight=0, bias=0, ): #initialises the attributes of the class with adjustable learning rate and epochs 
        self.weight = weight              #stores weight 
        self.bias = bias                  #stores bias 
        self.x = []                       #creates empty list to store our predictor variables (x)  
        self.y = []                       #creates empty list to store our our predicted variables (y) 
        self.learning_rate = learning_rate #setting your own learning rate 
        self.epochs = int(epochs)               #setting your own number of epochs 
 
    def vectorise_x(self, x):                   #method to store, convert and display error messages for input data 
        if isinstance(x, (pd.DataFrame)):       #if input data is as dataframe display error message to request series 
            print("Error: Select features to create pandas series")
        elif isinstance(x, (tuple, list, np.ndarray)): #valid input data as the type of tuples, lists, arrays that are numeric, iterable and indexable 
            self.x = list(x)                           #convert these valid types into lists 
            print("Ready to go!")
        elif isinstance(x, (pd.Series)):               #method to handle panda series 
            self.x = list(x.values)                    #extract the values from the pandas series 
            print("Ready to go!")
        else:
            print("Error: Please insert iteratable, numeric type like `tuple` or `pd.Series`") #any other data type displays error message 
                      
    def vectorise_y(self, y):
        if isinstance(y, (pd.DataFrame)):
            print("Error: Select features to create pandas series")
        elif isinstance(y, (tuple, list, np.ndarray)): 
            self.y = list(y)
            print("Ready to go!")
        elif isinstance(y, (pd.Series)):
            self.y = list(y.values)
            print("Ready to go!")
        else:
            print("Error: Please insert iteratable, numeric type like `tuple` or `pd.Series`")
        
                      

    def predict_y (self):                 #calculating the predicted y[i] for our optimisation later 
        self.y_predict =[]                #creating an empty list to store all predicted y values 
        n = len(self.y)                   #range that we iterate over (number of values of y)

        for i in range(n):                #looping over the number of values we have in the dataset 
            self.y_predict.append(self.weight*self.x[i] + self.bias)     #calculating predicted y values with line equation and adding predicted values to our list 
        return self.y_predict                             

#NUMERICAL OPTIMISATION 
#Creating method to get weight
    def partial_w(self):                       #partial derivative in regard to weight 
        self.y_predict = self.predict_y()      #predicted y value is equal to calling the internal method we defined above 
        gradient = 0 
        n=len(self.y)

        for i in range(n):
            gradient += self.x[i]*(self.y_predict[i] - self.y[i])    #partial derivative equation to calculate total partial derivative of weight in regards to error function
        return (-2/n)*gradient                                       #returns the  weight eqaution that minimises the partial derivative in regard to error function

#Creating method to get bias 
    def partial_b(self):
       n=len(self.y)
       gradient = 0
       self.y_predict = self.predict_y()
       
       for i in range(n):
            gradient += (self.y_predict[i]- self.y[i])               #partial derivative equation to calculate total partial derivative of bias in regards to error function
       return (-2/n)*gradient                                        #returns the  bias equation that minimises the partial derivative in regard to error function

#Gradient Descent - iterating over multiple steps with our partial weight and bias functions 
    def optimise(self): 
        learn_rate = self.learning_rate                #size of steps we make "downhill" to minimise total error in regards to the weight and bias 

        for i in range(self.epochs):                   #number of "epochs"/ steps we take in order to minimise aggregate error 
            self.weight = self.weight + learn_rate * self.partial_w() #optimised weight by calling partial_w as many times as epochs entered
            self.bias = self.bias + learn_rate * self.partial_b()     #optimised bias  by calling partial_b as many times as epochs entered
            if i % 10 == 0:                                           #prints out the weight and bias every 10 epochs 
                print( f"Weight: {self.weight} , Bias: {self.bias}" )

#DISPLAYING MODEL INTERNALS    
#Residuals - creating a new residuals method to display deviation of predicted values from actual values
    def residuals(self):
        self.residuals_list = []
        n=len(self.x) 

        for i in range(n):
            self.residuals_list.append(self.y[i] - (self.weight * self.x[i] + self.bias)) #adding to the list called "residuals" the difference between actual and predicted y
        
        for i in range(n):
            print(float(self.residuals_list[i]))
    
    def plot_residuals(self):
        plt.scatter(self.x, self.residuals_list)
        plt.axhline(y = 0, color = 'red')
        plt.title("Plotted residuals")
        plt.xlabel("Independent variable")
        plt.ylabel("Residuals")

#Display values of self.x, self.y and predicted y and plot residuals 
    def display_x(self): #displays all of the independent variables 
        n = len(self.x)

        for i in range(n):
            print(float(self.x[i]))

    def display_y(self): #displays all of the dependent variables 
        n = len(self.y)

        for i in range(n):
            print(float(self.y[i]))

    def display_weight(self):
        return float(self.weight)
    
    def display_bias(self):
        return float(self.bias)
                                                                                   
#EVALUATION METRICS  -  these are key values that we will use to quantify how good our model predicts the data it is trained on. 
#Mean Squared Error (MSE)  - the average squared deviation from actual values of y

    def mse(self):
        mse = 0                              #initialising our mse as a variable  that will be updated through the loops 
        n=len(self.y)                        #creating length for range to iterate over
        square_error = 0                     #initialising the sqaure error aas zero 

        for i in range(n):                   #iterating to calculate the mse 
            square_error += ((self.y[i] - (self.weight * self.x[i] + self.bias))**2)
        mse = square_error/n                 #calculates mse by dividing sqaure_error by n 
        return float(mse)
            
#R^2 -  how much of the deviation in y is explained by our model
    def rsquared(self):
        n=len(self.y)
        self.avg_y = 0                       #initial value of the average of our actual y values 

    #Average y- average of our actual y    
        for i in range(n):
            self.avg_y += ((1/n)*self.y[i])  #calculating the average value of actual y 
        
    #Total sum of squares   
        self.sum_squares = 0                 #creating an object called sum_squares to be used further in the function 
        sum_squares_list =[]                 #empty list to store values of sum of squares 
        n = len(self.y)

        for i in range(n):
            sum_squares_list.append((self.y[i] - self.avg_y)**2) #the squared values of actual - predicted y  and storing them in the empty list above 
            self.sum_squares += sum_squares_list[i]              #adding togther all of the sum of squares into initial variable sum_squares 
    #Squared error 
        self.square_error = 0                

        for i in range(n):
            self.square_error +=((self.y[i] - (self.weight * self.x[i] + self.bias))**2)                           
    #Final calculation 
        rsquared = 0                         #initialising our value of rsquared as 0 
        n=len(self.y)

        rsquared = (1-(self.square_error/self.sum_squares)) #calculating R^2 with our instances of sum of squares and square error 
        return float(rsquared) 

#Predictions - method to predict dependent variable based on an input independent variable of interest
    def predict(self, x): 
        if isinstance(x, (pd.DataFrame)):       #if input data is as dataframe display error message to request series 
            print("Error: Use iterable without metadata like 'pd.Series`")

        elif isinstance(x, (tuple, list, np.ndarray)): #valid input data as the type of tuples, lists, arrays that are numeric, iterable and indexable 
            m = len(x) 
            predict =[]
            for i in range(m):
                predict.append(self.weight * x[i] + self.bias)
            for i in range(m):
                print(float(predict[i]))
            
        elif isinstance(x, (pd.Series)):               #method to handle panda series 
            m = len(x) 
            predict =[]
            for i in range(m):
                predict.append(self.weight * x.iloc[i] + self.bias)
            for i in range(m):
                 print(float(predict[i]))
            
        elif isinstance(x, (int, float)):
            predict = self.weight * x + self.bias
            print(float(predict))

