import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle
from pymongo import MongoClient
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold

def train_rf_model(coll_chi,coll_lon,coll_sf):
    # load data from the database
    df_chi = pd.DataFrame(list(coll_chi.find()))
    df_lon = pd.DataFrame(list(coll_lon.find()))
    df_sf  = pd.DataFrame(list(coll_sf.find()))

    df = pd.concat([df_chi,df_lon,df_sf])

    # get the float values out of the json_data
    df_features = df.features.apply(json.loads)
    df_features = df_features.apply(json_normalize)
    df_features = pd.concat(df_features.tolist())

    # get train and test data
    x_train, x_test, y_train, y_test = train_test_split(df_features.values, df.label_val.values)

    # Create model
    log_model = LogisticRegression()
    rf_model = RandomForestClassifier(n_estimators = 3000,
                                      max_features = 'sqrt',
                                      bootstrap    = True,
                                      oob_score    = True,
                                      n_jobs       = -1,
                                      verbose      = 1,
                                      )

    # fit model
    log_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)

    with open("data/tmp_log_model.pkl", 'w') as f:
        pickle.dump(log_model, f)
    with open("data/tmp_rf_model.pkl", 'w') as f:
        pickle.dump(rf_model, f)
    # save model

    # run test set through model.
    log_pred_y = log_model.predict(test_x)
    rf_pred_y = rf_model.predict(test_x)

    # get scores
    print log_model.score(x_test, y_test)
    print rf_model.score(x_test, y_test)



if __name__ == '__main__':
    #******************************* THIS SCRIPT TO OPEN DB ************************
        DB_NAME = 'TRAINING_FEATURES'
        client = MongoClient()
        db = client[DB_NAME]
        coll_chi = db['Chicago']    # connect to mongodb to store scraped data
        coll_lon = db['London']    # connect to mongodb to store scraped data
        coll_sf = db['San_Francisco']    # connect to mongodb to store scraped data

        train_rf_model(coll_chi,coll_lon,coll_sf)
