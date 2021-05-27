# MLDLS
Machine Learning and Deep Learning Show 2021

Assignment 1 : Linear Regression (Code Explanation)

## Importing Libraries

Importing a library means loading it into the memory and then it’s there for us to work with.

•	NumPy stands for Numerical Python. NumPy Library is used for working with arrays.

•	Pandas stands for “Python Data Analysis Library”. It is used to import data from local storage usually CSV file. In short Pandas is used for managing datasets.

•	Matplotlib Library helps us visualising data into plot/graphs. It helps us customize our plots.

## Importing Dataset

Dataset can be imported using functions of Pandas which is read_csv which reads comma-separated values (csv) files and creates a DataFrame (represents data in tabular form with labelled rows and columns).

Code: *df = pd.read_csv(“Sample.csv”)*

## iloc( ) function

The iloc( ) function helps us select a value that belongs to a particular row or column from a set of values of a dataset. It considers only integer indexes as parameters.

Syntax: *iloc [ rows, columns] i.e. more precisely (iloc[starting index of row: ending index of row, starting index of column: ending index of column])*

Example:

•	iloc [ : , : ] --- All rows, all columns 

•	iloc [ : , 1] --- All rows, 2nd columns 

•	iloc [ : , 0] --- All rows, 1st Column

•	iloc [: , -1] --- All rows, last column

Next, we will plot the data into visuals using matplotlib.pyplot and label the plot for better understanding. We use scatter function to represent each and every datapoint separately.

## Splitting the dataset

To train any machine learning model we have to split the dataset into training data and testing data. To split the data we import ‘train_test_split’ function from submodule sklearn library that is model_selection.

Code: 

*from sklearn.model_selection import train_test_split*

*x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)*

The following parameters in ‘train_test_split’:

1)	x and y had previously defined
2)	test_size: This is set 0.2 thus defining the test size will be 20% of the dataset

## Fitting Linear Regression Model

First we will import LinearRegression from sklearn(Scikit-Learn) Library. Then using the fit() function, to fit our Model(Linear Regresion). The fit() method takes the training data as arguments.

To test what the model learned from the training data we use predict( )function.

predict() function enables us to predict the labels of the data values on the basis of the trained model. The predict() function accepts only a single argument which is usually the data to be tested.

## Accuracy Calculation

Accuracy is a measure of how accurate the model is. 

Accuracy = Correct Prediction / Total Prediction

For Regression Model , accuracy is calculated using formula called Root Mean Square Error (RMSE).

Linear Regression, y=mx +c 

Cost Function (error/performance of the equation) = *1/2m Sum i to m [(y'-y)^2]*

While writing program for our model , we use score( ) function to get the accuracy of the model. 

Code: *accuracy = model.score(X_test, Y_test)*
