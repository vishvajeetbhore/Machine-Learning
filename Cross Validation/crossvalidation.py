from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
 
# Load the data set
data = np.loadtxt('polynome.data')

X = data[:, 0]
Y = data[:, 1]
N = len(X)

def visualize(w):
    # Plot the data
    plt.plot(X, Y, 'r.')
    # Plot the fitted curve 
    x = np.linspace(0., 1., 100)
    y = np.polyval(w, x)
    plt.plot(x, y, 'g-')
    plt.title('Polynomial regression with order ' + str(len(w)-1))
    plt.show()

# Apply polynomial regression of order 2 on the data
#w = np.polyfit(X, Y, 7)
'''
for deg in range(1,11):
    X_train = X[:11]
    X_test = X[11:]
    Y_train = Y[:11]
    Y_test = Y[11:]
    w = np.polyfit(X_train, Y_train, deg)
    predict = np.polyval(w, X_test)
    error = np.sum((Y_test - predict)**2)/(2 * N)
    #print(deg,error)
    #visualize(w)
'''
# Visualize the fit

print('kfold  cross validation')
k = 1
nb_split = int(N/k)

X_sets = np.hsplit(X, nb_split)
Y_sets = np.hsplit(Y, nb_split)
for deg in range(1, 11):
    train_error = 0.0
    test_error = 0.0
    for s in range(nb_split):
        X_train = np.hstack(X_sets[1] for i in range(nb_split) if not i == s)
        print(X_train)
        Y_train = np.hstack(Y_sets[1] for i in range(nb_split) if not i == s)
        #w = np.polyfit(X_train, Y_train, deg)
        Y_fit_train = np.polyval(w, X_train)
        Y_fit_test  = np.polyval(w, X_sets[s])
        train_error += .5 * np.dot((Y_train - Y_fit_train).T, (Y_train - Y_fit_train)) / float(nb_split)
        test_error += .5 * np.dot((Y_sets[s] - Y_fit_test).T, (Y_sets[s] - Y_fit_test)) / float(nb_split)
        #print(deg, train_error, test_error)