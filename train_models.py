# ********************************************************************************
# The functions in this file are for exploring models for making predictions on
# a dataset featurized through a vgg neural net.
# ********************************************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import multiprocessing

def get_data():
    '''
    get the data which is stored on disk.
    Return array of training features (each row an image), and an array of labels
    '''
    prefix = 'data/'
    df_list = []
    for i in xrange(1000,10000,1000):
        df_chi = pd.read_pickle(prefix+'Chicago'+str(i)+'.pkl')
        df_lon = pd.read_pickle(prefix+'London'+str(i)+'.pkl')
        df_sf = pd.read_pickle(prefix+'San_Francisco'+str(i)+'.pkl')
        df_sf['label_val'] = 0  # make sure ONE label is zero until I train a "none of the above"
        df_list.append(df_sf)
        df_list.append(df_chi)
        df_list.append(df_lon)

    df = pd.concat(df_list)

    # put data in arrays
    X = df.drop(['label', 'label_val','url'], axis=1).values
    y = df['label_val'].values

    return X, y

def train_svm_model():
    '''
    Train an svm model using the featurized data.
    Write model to file.
    Print score to console.
    '''
    # Load data from pickle files
    X, y = get_data()

    # get train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # Create model
    # log_model = LogisticRegression(n_jobs=-1, verbose=1)
    svm_model = SVC(
                    kernel = 'rbf',
                    gamma = .001,
                    verbose = True,
                    decision_function_shape='ovr',
                    C=10,
                    probability=True
                    )

    # fit model
    # get scores
    # Current accuracy: tmp_svm_model1--> .73, C=1
    # .53, C=.01
    # .63, C=.1
    # .74:  C=10, gamma=.001 , rbf
    svm_model.fit(x_train, y_train)

    # get scores
    # Current accuracy: tmp_svm_model1--> .73
    print "SVM: C=10", svm_model.score(x_test, y_test)

    # save model
    with open("/data/tmp_svm_model2.pkl", 'w') as f:
        pickle.dump(svm_model, f)


def run_svm_model(args_tup):
    '''
    ***** Function designed to play well with multiprocessing. *****
    Input:  tuple containing the parameters for an svm model as a dictionary.
    Create and fit an svm model with the input parameters.
    Output: the model parameters as a dictionary, the accuracy score.
    '''
    # Load data from pickle files
    params = args_tup[0]

    X, y = get_data()
    # get train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # Create model
    svm_model = SVC(**params)

    svm_model.fit(x_train, y_train)

    score = svm_model.score(x_test, y_test)

    return params, score


def train_svm_model_parallel():
    '''
    Make a list of dictionaries of different parameter settings for svm models.
    Pass parameters to svm models and run in parallel using multiprocessing.
    Print the parameters and accuracy score for each result to console.
    Pickle the best set of parameters to save for later.
    '''
    # Create Parameter dictionareis
    params1 = {
                'kernel':'rbf',
                'gamma':.001,
                'decision_function_shape':'ovr',
                'C':30
            }
    params2 = {
                'kernel':'rbf',
                'gamma':.0001,
                'decision_function_shape':'ovr',
                'C':30
            }
    params3 = {
                'kernel':'poly',
                'degree':3,
                'gamma':.001,
                'decision_function_shape':'ovr',
                'C':10
            }
    params4 = {
                'kernel':'poly',
                'degree':3,
                'gamma':.0001,
                'decision_function_shape':'ovr',
                'C':10
            }
    params5 = {
                'kernel':'poly',
                'degree':3,
                'gamma':.001,
                'decision_function_shape':'ovr',
                'C':30
            }
    params6 = {
                'kernel':'poly',
                'degree':3,
                'gamma':.0001,
                'decision_function_shape':'ovr',
                'C':30
            }


    params_list = [params1, params2, params3, params4, params5, params6]

    # create svm models for each parameter dict in parallel.
    pool = multiprocessing.Pool(6)
    results_list = pool.map(run_svm_model, [(params,) for params in params_list])

    best_score = 0
    for results in results_list:
        print results[0]
        print '*** Score: ', results[1]
        print
        if results[1]> best_score:
            best_score = results[0]
            best_params = results[0]

    # save best result
    # best so far:  poly 3, c:1, score: .72; rbf, C:10, gamma: auto
    with open('data/best_svm_params.pkl','w') as f:
        pickle.dump(best_params)


def train_rf_model():
    '''
    Train a reandom forest model using the featurized data.
    Write model to file.
    Print score to console.
    '''
    # Load data from pickle files
    X, y = get_data()

    # Create model
    # log_model = LogisticRegression(n_jobs=-1, verbose=1)
    rf_model = RandomForestClassifier(n_estimators = 4000,
                                      max_features = 'sqrt',
                                      max_depth = None,
                                      bootstrap    = True,
                                      oob_score    = True,
                                      n_jobs       = -1,
                                      verbose      = 1,
                                      )

    # fit model; No need for train-test split.
    # Use out of bag score instead.
    rf_model.fit(X, y)

    # get out of bag score
    print rf_model.oob_score_


    # save model
    with open("/data/tmp_rf_model2.pkl", 'w') as f:
        pickle.dump(rf_model, f)

def train_logistic_model():
    '''
    Train a logistic regrssion model using the featurized data.
    Write model to file.
    Print score to console.
    '''
    # Load data from pickle files
    X, y = get_data()

    # get train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # Create model
    log_model = LogisticRegression(n_jobs=-1, verbose=1)


    # fit model
    log_model.fit(x_train, y_train)

    # get scores
    print log_model.score(x_test, y_test)

    # save model
    with open("/data/tmp_log_model.pkl", 'w') as f:
        pickle.dump(rf_model, f)



def explore_model():
    '''
    '''
    # Load data from pickle files
    X, y = get_data()

    ######### Uncoment to explore Random Forest Classifier ################
    # rf_params = {'max_features':['sqrt',None],
    #             'bootstrap': [True],
    #             'n_estimators':[3000,4000,5000],
    #             'max_depth':[None, 4, 6],
    #             'min_samples_leaf':[1,5,9]
    #             }
    # rf_grid = GridSearchCV(RandomForestClassifier(),rf_params, cv=5, n_jobs=-1,verbose=1)
    # rf_grid.fit(X, y)
    #
    # print "Best score: ",rf_grid.best_score_
    # print "Best params: ",rf_grid.best_params_
    #
    # with open('data/best_params_rf.txt','w') as f:
    #     f.write(rf_grid.best_params_)
    ############################################################################

    ################### Explore SVM classifier #################################
    svc_params = {'C':[.01,.1,10]
                }
    svc_grid = GridSearchCV(SVC(kernel = 'rbf',
                               gamma = 'auto',
                               verbose = False,
                               decision_function_shape='ovr'),
                           svc_params, cv=5, n_jobs=6,verbose=2)
    svc_grid.fit(X, y)

    print "Best score: ",svc_grid.best_score_
    print "Best params: ",svc_grid.best_params_

    with open('data/best_params_svc.txt','w') as f:
        f.write(rf_grid.best_params_)
    ############################################################################

if __name__ == '__main__':

    # train_svm_model()
    # explore_model()
    train_svm_model_parallel()

    ################ SCORES SCORES SCORES #############################
# {'kernel': 'rbf', 'C': 10, 'decision_function_shape': 'ovr', 'gamma': 0.1}
# *** Score:  0.331901022374
#
# {'kernel': 'rbf', 'C': 10, 'decision_function_shape': 'ovr', 'gamma': 0.01}
# *** Score:  0.347310712698
#
# {'kernel': 'rbf', 'C': 10, 'decision_function_shape': 'ovr', 'gamma': 0.001}
# *** Score:  0.739516965476
#
# {'kernel': 'rbf', 'C': 10, 'decision_function_shape': 'ovr', 'gamma': 0.0001}
# *** Score:  0.7143280486
#
# {'kernel': 'poly', 'C': 1, 'decision_function_shape': 'ovr', 'gamma': 0.001, 'degree': 3}
# *** Score:  0.732108460513
#
# {'kernel': 'poly', 'C': 1, 'decision_function_shape': 'ovr', 'gamma': 0.0001, 'degree': 3}
# *** Score:  0.643947251445
# {'kernel': 'rbf', 'C': 30, 'decision_function_shape': 'ovr', 'gamma': 0.001}
# *** Score:  0.73255297081
#
# {'kernel': 'rbf', 'C': 30, 'decision_function_shape': 'ovr', 'gamma': 0.0001}
# *** Score:  0.709882945622
#
# {'kernel': 'poly', 'C': 10, 'decision_function_shape': 'ovr', 'gamma': 0.001, 'd
# egree': 3}
# *** Score:  0.73255297081
#
# {'kernel': 'poly', 'C': 10, 'decision_function_shape': 'ovr', 'gamma': 0.0001, '
# degree': 3}
# *** Score:  0.712994517706
#
# {'kernel': 'poly', 'C': 30, 'decision_function_shape': 'ovr', 'gamma': 0.001, 'd
# egree': 3}
# *** Score:  0.73255297081
#
# {'kernel': 'poly', 'C': 30, 'decision_function_shape': 'ovr', 'gamma': 0.0001, '
# degree': 3}
# *** Score:  0.737146243888
