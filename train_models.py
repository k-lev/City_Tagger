import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

def train_rf_model():
    # load data from the database
    # df_chi = pd.DataFrame(list(coll_chi.find()))
    # df_lon = pd.DataFrame(list(coll_lon.find()))
    # df_sf  = pd.DataFrame(list(coll_sf.find()))

    # # get the float values out of the json_data
    # df_features = df.features.apply(json.loads)
    # df_features = df_features.apply(json_normalize)
    # df_features = pd.concat(df_features.tolist())

    # Load data from pickle files
    prefix = '/data/'
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

    # get train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # Create model
    # log_model = LogisticRegression(n_jobs=-1, verbose=1)
    rf_model = RandomForestClassifier(n_estimators = 3000,
                                      max_features = 'sqrt',
                                      bootstrap    = True,
                                      oob_score    = True,
                                      n_jobs       = -1,
                                      verbose      = 1,
                                      )

    # fit model
    # log_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)

    # with open("/data/tmp_log_model.pkl", 'w') as f:
        # pickle.dump(log_model, f)
    with open("/data/tmp_rf_model.pkl", 'w') as f:
        pickle.dump(rf_model, f)
    # save model

    # run test set through model.
    # log_pred_y = log_model.predict(test_x)
    # rf_pred_y = rf_model.predict(x_test)

    # get scores
    # print log_model.score(x_test, y_test)
    print rf_model.score(x_test, y_test)

def explore_model():
    '''
    '''
    # Load data from pickle files
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

    rf_params = {'max_features':['sqrt',None],
                'bootstrap': [True],
                'n_estimators':[3000,4000,5000],
                'max_depth':[None, 4, 6],
                'min_samples_leaf':[1,5,9]
                }
    rf_grid = GridSearchCV(RandomForestClassifier(),rf_params, cv=5, n_jobs=-1,verbose=1)
    rf_grid.fit(X, y)

    print "Best score: ",rf_grid.best_score_
    print "Best params: ",rf_grid.best_params_

    with open('data/best_params_rf.txt','w') as f:
        f.write(rf_grid.best_params_)


if __name__ == '__main__':
    # #******************************* THIS SCRIPT TO OPEN DB ************************
    #     DB_NAME = 'TRAINING_FEATURES'
    #     client = MongoClient()
    #     db = client[DB_NAME]
    #     coll_chi = db['Chicago']    # connect to mongodb to store scraped data
    #     coll_lon = db['London']    # connect to mongodb to store scraped data
    #     coll_sf = db['San_Francisco']    # connect to mongodb to store scraped data

    # train_rf_model()
    explore_model()
