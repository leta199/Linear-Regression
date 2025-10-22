# Linear-Regression
Linear regression created from scratch and comparing my implementation to that of the popular packages  scikit -learn. 


## HOW IT'S MADE 

Languages used: Python verion (3.13.2)  
Packages and modules: numpy, pandas, matplotlib, scikit-learn  
Environment: VScode  

## METHOD 
### CLASS CREATION  
To create this model I started by creating a class in Python to store the entire model logic inclusing the model's attributes and methods.  
These were split up into the main areas of:
- Initialisation of the class and arguments
- Method to take in modelling data points
- Numerical optimisation via gradient descent
- Displaying key internals of the model such as predictions
- Evaluation metrics common to linear regression such as Mean squared error

**Initialisation**  
I initialised the class using the intialiser  `__init__`  as well as defined the arguments of the class as:
1) learning rate
2) number of epochs
3) weight and bias is set at 0

**Method to input data points**   
This method called `vectorise()` takes in the points to be modelled and makes sure that they are:
- Numeric, iteratable and indexable (e.g tuples, lists and np.arrays) - if so they are converted into a list
If they are not numeric, indexeable or iteratable like pd.Dataframes they are convtrted into lists with only the values and removinf any metadata

Return appropriate errors if any data is entered  in the incorrect format like singular value.

 **Numerical optimisation**
The actual group of  methods that calculate the optimal weight and bias of the data.  
`preditc_y()` - an internal method to caluculate the predicted y value in the model.  
`partial_w()` - calculates the weight that minimises our partial derivative of error in regard to the weight to get the global minimum.
`partial_b()` -calculates the bias that minimises our partial derivative of error in regard to the bias to get the global minimum.
`optimise()` - uses the number of epochs and iterates over that many epochs while taking steps towrds global minimum in regard to weight and bias at once in the sixe of the  learning rate. Also prints these weights and biases every 10 epochs to see how the optimisation is progressing 

**Displaying model internals**   
`residuals()` - appends the residuals of the model (difference between  the predicted and actual output) into a list to display tp users.  
`display_x()` ,`display_y()` - methods that display the entered x and y data as lists from the class  
`display_predict()` - displays the predicted y values based on our optimised weight and bias i.e on line of best fit. 









### Data  
https://catalog.data.gov/dataset/summary-metrics-by-decile-and-quartile
This data was collected from the SEC  redarding data about stock prices grouped by decile and market capitalisation. I made sure to focus primarily 
### Linear regression model 


 ## PROJECT STRUCTURE      
[Linear-Regression](https://github.com/leta199/Linear-Regression)  
├──[Finance application](https://github.com/leta199/Linear-Regression/tree/main/finance_application)  
│  ├──[Linear Regression Finance](https://github.com/leta199/Linear-Regression/blob/main/finance_application/Linear_Regression_Finance.ipynb)   
│  └──[Finance data](https://github.com/leta199/Linear-Regression/blob/main/finance_application/finance_data.csv)  
│  
├──[Model](https://github.com/leta199/Linear-Regression/tree/main/model)  
│  ├──[Linear Regression (core model)](https://github.com/leta199/Linear-Regression/blob/main/model/Linear_Regression.ipynb)  
│  └──[Mathematical logic](https://github.com/leta199/Linear-Regression/blob/main/model/Mathematical%20logic.pdf)  
│  
├──[License](https://github.com/leta199/Linear-Regression/blob/main/LICENSE)  
└──[README](https://github.com/leta199/Linear-Regression/blob/main/README.md)


## USEFUL EDUCATIONAL RESOURCES    
[Winsorized mean](https://www.datacamp.com/tutorial/winsorized-mean)   
[Dealing with outliers](https://www.analyticsvidhya.com/blog/2022/09/dealing-with-outliers-using-the-iqr-method/)

"This is an educational implementation of linear regression for learning purposes — not intended for production use."
## LOOKING TO THE FUTURE  
1) I wish to extend my model to be able to work on all kinds of interatables like pandas series and numpy arrays.
2) Add methods to adjust learning rate and number of epochs.
3) Add method to show evaluation metrics like R squared, Mean squared error. 
4) Add methods to display internal parametrs like the learned weight and bias, residuals and predicted values.
5) Display error messages when taking in inputs using the `vectorise()` method.
   
## AUTHORS 
[leta199](https://github.com/leta199)
