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

    # Load data from pickle files
    X, y = get_data()

    # get train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # Create model
    # log_model = LogisticRegression(n_jobs=-1, verbose=1)
    svm_model = SVC(
                    kernel = 'rbf',
                    gamma = 'auto',
                    verbose = True,
                    decision_function_shape='ovr',
                    C=10
                    )

    # fit model
    # get scores
    # Current accuracy: tmp_svm_model1--> .73, C=1
    # .53, C=.01
    # .63, C=.1
    svm_model.fit(x_train, y_train)

    # get scores
    # Current accuracy: tmp_svm_model1--> .73
    print "SVM: C=.01", svm_model.score(x_test, y_test)

    # save model
    with open("data/tmp_svm_model2.pkl", 'w') as f:
        pickle.dump(svm_model, f)

def run_svm_model(args_tup):
    # Load data from pickle files
    params = args_tup[0]

    X, y = get_data()
    # get train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # Create model
    svm_model = SVC(params)

    svm_model.fit(x_train, y_train)

    score = svm_model.score(x_test, y_test)

    return params, score


def train_svm_model_parallel():

    # Load data from pickle files
    X, y = get_data()

    # get train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # Create model
    params1 = {
                'kernel':'rbf',
                'gamma':'auto',
                'decision_function_shape':'ovr',
                'C':10
            }
    params2 = {
                'kernel':'poly',
                'degree':2,
                'gamma':'auto',
                'decision_function_shape':'ovr',
                'C':1
            }
    params3 = {
                'kernel':'poly',
                'degree':3,
                'gamma':'auto',
                'decision_function_shape':'ovr',
                'C':1
            }
    params4 = {
                'kernel':'poly',
                'degree':2,
                'gamma':'auto',
                'decision_function_shape':'ovr',
                'C':.1
            }
    params5 = {
                'kernel':'poly',
                'degree':3,
                'gamma':'auto',
                'decision_function_shape':'ovr',
                'C':.1
            }


    params_list = [params1, params2, params3, params4, params5]

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

    with open('data/best_svm_params.pkl') as f:
        pickle.dump(best_params)


def train_rf_model():
    # Load data from pickle files
    X, y = get_data()

    # get train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y)

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

    # fit model
    # log_model.fit(x_train, y_train)
    # rf_model.fit(x_train, y_train)
    rf_model.fit(X, y)

    # get scores
    # print log_model.score(x_test, y_test)
    # print rf_model.score(x_test, y_test)
    print rf_model.oob_score_


    # save model
        # with open("/data/tmp_log_model.pkl", 'w') as f:
        # pickle.dump(log_model, f)
    with open("/data/tmp_rf_model2.pkl", 'w') as f:
        pickle.dump(rf_model, f)

    # run test set through model.
    # log_pred_y = log_model.predict(test_x)
    # rf_pred_y = rf_model.predict(x_test)



def explore_model():
    '''
    '''
    # Load data from pickle files
    X, y = get_data()

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

if __name__ == '__main__':
    # #******************************* THIS SCRIPT TO OPEN DB ************************
    #     DB_NAME = 'TRAINING_FEATURES'
    #     client = MongoClient()
    #     db = client[DB_NAME]
    #     coll_chi = db['Chicago']    # connect to mongodb to store scraped data
    #     coll_lon = db['London']    # connect to mongodb to store scraped data
    #     coll_sf = db['San_Francisco']    # connect to mongodb to store scraped data

    train_svm_model()
    # explore_model()
    # train_svm_model_parallel()
