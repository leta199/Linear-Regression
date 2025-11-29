# Linear-Regression
Built a linear regression model from scratch in Python to understand the mechanics behind predictive modeling using SEC market data. Implemented gradient descent, feature scaling, and performance evaluation, then compared results with scikit-learn’s version.   
The project blends hands-on coding with applied quantitative reasoning and real-world financial data analysis.

The notebook will:
- Load and clean SEC financial data from the `/data`.
- Import and train a custom regression model from `/model/LinearRegression.py`.
- Generate predictions and visualizations.
- Compare results with scikit-learn’s built-in LinearRegression model.

## HOW IT'S MADE 

Languages used: Python  
Packages and modules: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pathlib`  
Environment: VScode  

![Python](https://img.shields.io/badge/Python-3.13.5-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-lightgrey)

## LINEAR REGRESSION MODEL  
### METHOD 
To create this model I started by creating a class in Python to store the entire model logic inclusing the model's attributes and methods.  
These were split up into the main areas of:

- Initialisation of the class and arguments.
- Method to take in modelling data points.
- Numerical optimisation via gradient descent.
- Display model residuals and plot residuals. 
- Displaying key internals of the model such as weight and bias. 
- Evaluation metrics common to linear regression such as Mean squared error and R squared.
- Method to predict dependent variable. 


**Initialisation**  
I initialised the class using the intialiser  `__init__`  as well as defined the arguments of the class as:
1) learning rate
2) number of epochs
3) weight and bias is set at 0

**Method to input data points**   
This method called `vectorise()` takes in the points to be modelled and makes sure that they are:
- Numeric, iteratable and indexable (e.g tuples, lists and np.arrays) - if so they are converted into a list.
If they are not numeric, indexeable or iteratable like pd.Dataframes they are converted into lists with only the values and removing any metadata.

Return appropriate errors if any data is entered  in the incorrect format like singular values or non-numeric types.

 **Numerical optimisation**  
 The actual group of  methods that calculate the optimal weight and bias of the data.  
`preditc_y()` - an internal method to calculate the predicted y value in the model.  
`partial_w()` - calculates the weight that minimises our partial derivative of error in regard to the weight to get the global minimum.
`partial_b()` -calculates the bias that minimises our partial derivative of error in regard to the bias to get the global minimum.
`optimise()` - uses the number of epochs and iterates over that many epochs while taking steps towrds global minimum in regard to weight and bias at once in the direction of the  learning rate. Also prints these weights and biases every 10 epochs to see how the optimisation is progressing. 

**Displaying Residuals and Visualising Residuals**
`residuals()` - appends the residuals of the model (difference between  the predicted and actual output) into a list to display to users.  

`plot_residuals()` plots residuals against the independent variable x to see the distribution of residuals.   
The expected residuals for a good linear model with approporaite data that fits all the assumptions is a random cloud with no trend or "fan" shape.    
An example from synthetic data: 

<img width="578" height="459" alt="Image" src="https://github.com/user-attachments/assets/0633fe2f-5e39-4880-bdfb-acd456d75886" />

**Displaying model internals**   
Displays stored values in the model that end users may interested in seeing.   

`display_x()` ,`display_y()` - methods that display the entered x and y data as lists from the class.  
`display_predict()` - displays the predicted y values based on our optimised weight and bias i.e on line of best fit. 
`display_weight()` - displays the optimised weight the model calculated with a simple `print()` statement.  
`display_bias()` - displays the optimised bias calculated via gradient descent.

**Evaluation metrics**   
`mse()`- calculates and displays the mean squared error of the model by calculating total square error and dividing by the number of data points to get the average.  
`rsquared()` - calculates the average of the input data points and subtracts this from the predicted y to get sum of squares, then calculated square error to final calculate the rsquared by dividng the two and subtracting them from 1.  

**Predictions**  
`predict()` - returns the predicted y value (independent variable) based on input x values(s). This method accepts values of x as tuples, lists, NumPy arrays, pandas Series, integers and floats. It also returns an error message if any other types are used for prediction such as pandas Dataframes. 

An exmaple from a pandas Series list of predictions is below:   
<img width="612" height="235" alt="Image" src="https://github.com/user-attachments/assets/8c16c7c7-2b8b-47e6-bf64-03b4fa6c1e5f" />

### COMPARISON TO SCIKIT-LEARN
In an effort to see the accuracy of my model and how well numerical optimisation via for loops works for linear regression.    

I began by importing `scikit-learn` as `sk` and then from the scikit-learn package importing one of their linear models LinearRegression as `lr()`.  

Use `np.set_seed()` to set a seed for reproducibility of the comparison.  
Generate random synthetic linear data with random noise to compare the two models to each other.  

Utilised my model with an appropraite learning rate (0.00015) and number of epochs (100000).  
Assigns the models weight, bias, mean squares error and R^2 to the variables:  
- `mine_mse`  
- `mine_weight`  
- `mine_bias`   
- `mine_rsquared`

Extracted the values from the synthetic data using the `.values()` method.  
Assigned these values to the training data set.  
Imported the model as `lr()`.  
Fit the model to the training data using the `fit()` method to generate model parameters.  
Assinged the rsquared to the variable `rsquared_train` on training data using the `.score()` method.  
Assigned the weight and bias to the variables `sk_coef` and `sk_intercept`.   
Generated the mean square error using the `mean_squared_error()` function and assigned mse to the variable `mse_train`.

#### Dataframe      
Created a dataframe to compare the my model to that of scikit-learn.  
-Displayed the key metrics, namely: mean squared error and R^2.   
-Displayed the calculated weight and bias from each model. 


## FINANCE APPLICATION 
To identify how well the model works, it was used on real life data from the Securitites and Exchange Commission. 

### Data  
[SEC Financial data](https://catalog.data.gov/dataset/summary-metrics-by-decile-and-quartile)  
This data was collected from the SEC  regarding data about stock prices grouped by decile and market capitalisation.

The main data file I utilised was the decile_cancel_to_trade csv file that aggregates companies by market capitalisation into 10 groups based on range of market capitalisation.  Each observation is a time-series organisation of each group of market capitalisation. 

Date- Date when the stock trade takes place.  
Market Cap 1 - 10 - This is the propotion of stocks proposed to trade that got cancelled organised Market capitalisation decile into 10 groups.  
Price Decile 1 - 10 - Average stock price of each decile from decile 1 to 10.   
Turnover Decile 1 - 10 - Number of stocks sold on that date relative to total number of stocks. 
Volatitliy Decile 1- 10 - The risk (variance) assocaited with each stock at that given date.

### METHOD 
Utilising the linear regression model on the financial data consisted of the following steps:  
1) Data discovery
2) Data cleaning and pre-processing
3) Modelling with linear regression


**Hypothesis**  
As the volatility in a stock grows, cancel to trade  in each market capitalisation grows as well. This may be because investors are more likely to not vommit to a trade when the stock is more volatile.  

**Data  discovery**  
Started with insallting all necessary depenedinces such as Pandas, NumPy and MatPlotLib.  
Analysed and inspected the data with the methods `.head()` and `.tail()`.   
Filtered data into just the market capitalisation cancel to trade metric and volatility in the same market capitalisation.  
Renamed cancel to trade metric (Market capitalisation Deciel  n) to  "Cancellation Rate" and (Volatility Decile n) to "Volatility" with `.rename()`.  

I looked over each market capitalisation and their cancellation rate vs volatility to make sure the assocaited data points fit the assumptions necessary to be modelled with linear regression, namely:
- linearity - does the data show a strong linear trend.
- homoscedaicty - does the data have consatnt vraince acroos all ranges of our independent variable.

This was done using scatter plots with `.scatter()`to see overall trend of data.  
Decile 1 market capitalisation did not have a very strong positive correlation.  
Decile 7 market capitalisation did  have a very strong positive correlation so I will use it in my modelling. 

**Data cleaning and pre-processing**
Utilised `.boxplot()` to identify outliers in the data which were found to be present only in the higher end of Cancellation rate and Volatility.  
This data accounted for 2% of both variables so it may have been removed however I found that since outliers are only in the higher ranges of the data for cancellation rate  so if it was removed this would represent removing systematic error. 
Therefore, the statistical properties and distribution of the data like homoscedacity would be affected particularly for cancellation rate.

Winsorisation -  Decided to utisilde winsoarisation on the outliers i.e capping them to the values of our Upper  and lower range to preserve all of the upper and lower data  points. I applied this both to the Cancellation rate (upper range)  and Volatility (upper and lower range).

Finally, plotted scatter plot and boxplots of the variables to ensure pre-processing worked well. 

**Linear Regression**   
Utilised my linear regression model to model the data from Decile 7. 
Cancel to trade metric renamed to (Cancellation rate) was the dependent variable.   
Volatility was our independent variable. 

<img width="566" height="453" alt="Image" src="https://github.com/user-attachments/assets/ff7463c2-d2f2-4a2d-bcee-3f3864ed0696" />


## FINAL INSIGHTS 
My original hypothesis was correct. **Volatility of stock prices does have a positive effect on cancellation of stock orders** .

**Interpretation of output**  
My model and scikit-learn managed to explain approxiamtely 70% of the variance in data as seen by both R-sqaured values. 

<img width="445" height="72" alt="Image" src="https://github.com/user-attachments/assets/e79cae63-5929-44b7-a47f-338613f44408" />


Weights were nearly identical and the weight in my model of 1.049. 
Biases differed between the models by 0.87.   
Mean squared error of my model was approximately 3.5 meaning that predictions of cancelattion rate were off by 3.5 orders on average.

## Setup & Installation 

Clone this repository and navigate into the project directory:

`git clone https://github.com/YourUsername/Linear-Regression.git`  
`cd Linear-Regression`

Create a virtual environment (recommended) and install dependencies:  
`python -m venv venv`  
`source venv/bin/activate`  
`venv\Scripts\activate`        

`pip install -r requirements.txt`

**Usage**

Once the environment is set up, launch Jupyter Notebook:  
`jupyter notebook`

Then open and run the notebook:  
`Linear_Regression_Finance.ipynb`

Note: this projects uses relative file paths therefore all imports such as of the `LinearRegression.py` file should be okay as long as file structure remains the same. 


## PROJECT STRUCTURE      
|[Linear-Regression](https://github.com/leta199/Linear-Regression)  
|├──[Finance application](https://github.com/leta199/Linear-Regression/tree/main/finance_application)  
│  ├──[Linear Regression Finance Application](https://github.com/leta199/Linear-Regression/blob/main/finance_application/Linear_Regression_Finance.ipynb)   
│  └──[Finance data](https://github.com/leta199/Linear-Regression/blob/main/finance_application/finance_data.csv)  
│    
|├──[model](https://github.com/leta199/Linear-Regression/tree/main/model)  
│  ├──[Linear regression.py](https://github.com/leta199/Linear-Regression/blob/main/model/LinearRegression.py)      
│  ├──[Linear Regression (core model)](https://github.com/leta199/Linear-Regression/blob/main/model/Linear_Regression.ipynb)      
│  ├──[Mathematical logic](https://github.com/leta199/Linear-Regression/blob/main/model/Mathematical%20logic.pdf)      
│  └──[__init__](https://github.com/leta199/Linear-Regression/blob/main/model/__init__.py)     
│    
|├──[data](https://github.com/leta199/Linear-Regression/tree/main/data)    
│  ├──[decile to cancel csv](https://github.com/leta199/Linear-Regression/blob/main/data/decile_cancel_to_trade_stock.csv)  
│  └──[requirements](https://github.com/leta199/Linear-Regression/blob/main/data/requirements.txt)  
|  
|├──[License](https://github.com/leta199/Linear-Regression/blob/main/LICENSE)  
|└──[README](https://github.com/leta199/Linear-Regression/blob/main/README.md)


## USEFUL EDUCATIONAL RESOURCES    
[Winsorized mean](https://www.datacamp.com/tutorial/winsorized-mean)   
[Dealing with outliers](https://www.analyticsvidhya.com/blog/2022/09/dealing-with-outliers-using-the-iqr-method/)  
[Linear regression from scratch](https://youtu.be/VmbA0pi2cRQ?si=DLD_hFtu1TFj-SMf)  
[Train - test splits](https://builtin.com/data-science/train-test-split)


"This is an educational implementation of linear regression for learning purposes — not intended for production use."

## LOOKING TO THE FUTURE  
1) I wish to extend my model to be able to work on all kinds of interatables like pandas series and numpy arrays ✅
2) Add methods to adjust learning rate and number of epochs ✅
3) Add method to display evaluation metrics like R squared, Mean squared error ✅ 
4) Add methods to display internal parameters like the learned weight and bias, residuals, predicted values  and internal values of x and y ✅
5) Display error messages when taking in inputs using the `vectorise()` method ✅
6) Method to plot residuals to make sure the data displays the "random cloud" we would expect from linear regression used appropriately ✅
7) Method to predict dependent variable for any value of independent variable entered ✅
8) Use linear model to model real financial data ✅
9) Use test - training splits and cross validation on future models .
   
## AUTHORS 
[leta199](https://github.com/leta199)
