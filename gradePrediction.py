# Import required libraries
import pandas
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read in data set
data = pandas.read_csv("student-mat.csv", sep=";")

# Obtain only required data values
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Define what the program will predict (In this case, final grade)
predict = "G3"

# Make a set without the predicted value
x = np.array(data.drop([predict], 1))

# Make a set containing only the predicted value
y = np.array(data[predict])

# Define variable to store best value
best = 0

# Obtain train and test values (put here for when you no longer are training the model)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)

# Train 30 times
for a in range(30):
    # Train the model using 90% of the given values, and save 10% to test with
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
    
    # Create linear model
    linear = linear_model.LinearRegression()

    # Find the best fit line and store it in linear
    linear.fit(x_train, y_train)

    # Determine the accuracy and output it
    acc = linear.score(x_test, y_test)

    # Store model if it is more accurate than the previous most accurate model
    if acc > best:
        best = acc
        print ("Current accuracy: " + str(acc))
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# Read in pickle file and load the model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Print out the coefficents used in calculation of predicted value
print("Coefficent: \n", linear.coef_)

# Print out the intercept of the predicted value (assuming all values were 0)
print("Intercept: \n", linear.intercept_)

# Store predictions
predictions = linear.predict(x_test)

# Output predictions given inputs, and the actual values
for x in range(len(predictions)):
    print ("input " + str(x+1) + ":")
    print ("predicted: ", predictions[x])
    print ("inputs: ", x_test[x])
    print ("actual value: ", y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
