import numpy as np

#-----------------------------------------------------------------------
# ROOT DECOMPOSITION FUNCTIONS

def split_f_root(X):
    X_split = {0: X[0][:,0:1],                 # past-load
               1: [X[0][:,1:5],X[1][:,0:4]],   # calendar & weather
               2: [X[0][:,5:15],X[1][:,4:14]]  # holiday
    }         
    return X_split

def joint_f_root(X_split):
    X_encoder = np.concatenate(
        [X_split[0], X_split[1][0], X_split[2][0]], 
        axis = 1
    ).reshape(180, 15)
    X_decoder = np.concatenate(
        [X_split[1][1],X_split[2][1]], 
        axis = 1
    ).reshape(36, 14)
    return [X_encoder,X_decoder]

#-----------------------------------------------------------------------
# ELEMENT 0 DECOMPOSITION FUNCTIONS

def split_f_0(X):
    X_split = {0: X[0][:-72,0:1],       # past-load 72-180 h before
               1: X[0][-72:-24,0:1],    # past-load 24-72 h before
               2: X[0][-24:-12,0:1],    # past-load 24-12 h before 
               3: X[0][-12:,0:1],       # past-load 1-12 h before
               'past': X[0][:,1:],      # past
               'future': X[1]           # future
    }         
    return X_split

def joint_f_0(X_split):
    X_past_load = np.concatenate(
        [X_split[0], X_split[1], X_split[2], X_split[3]], 
        axis = 0
    ).reshape(180, 1)
    X_encoder = np.concatenate(
        [X_past_load, X_split['past']], 
        axis = 1
    ).reshape(180, 15)
    return [X_encoder, X_split['future']]

#-----------------------------------------------------------------------
# ELEMENT 1 DECOMPOSITION FUNCTIONS

def split_f_1(X):
    X_split = {'load': X[0][:,0:1],      # fixed rest
               0: X[0][:-24,1:4],        # past-calendar 24-180 h before
               1: X[0][-24:,1:4],        # past-calendar 1-24 h before
               2: X[1][:,0:3],           # future-calendar
               3: X[0][:,4:5],           # past-weather
               4: X[1][:,3:4],           # future_weather
               'holiday': [X[0][:,5:15], X[1][:,4:14]] # fixed rest 
    }         
    return X_split

def joint_f_1(X_split):
    X_past_calendar = np.concatenate(
        [X_split[0], X_split[1]], 
        axis = 0
    ).reshape(180, 3)
    X_encoder = np.concatenate(
        [X_split['load'], X_past_calendar, 
         X_split[3], X_split['holiday'][0]], 
        axis = 1
    ).reshape(180, 15)
    X_decoder = np.concatenate(
        [X_split[2], X_split[4], X_split['holiday'][1]], 
        axis = 1
    ).reshape(36, 14)
    return [X_encoder,X_decoder]


#-----------------------------------------------------------------------
# Explainability hierarchical model

explain_dict ={
    "name": "Seq2seq LSTM model",
    "component_names": ["past-load", "calendar and weather", "holiday"],
    "joint_f": joint_f_root,
    "split_f": split_f_root,
    "components": [
        {
            "name": "past-load",
            "idx": 0,
            "component_names" : [
                "past-load 72-180 h before",
                "past-load 24-72 h before",
                "past-load 12-24 h before",
                "past-load 1-12 h before"
            ],
            "joint_f": joint_f_0,
            "split_f": split_f_0,
        },
        {
            "name": "calendar and weather",
            "idx": 1,
            "component_names" : [
                "past-calendar 24-180 h before",
                "past-calendar 1-24 h before",
                "future-calendar",
                "past-weather", 
                "future-weather"
            ],
            "joint_f": joint_f_1,
            "split_f": split_f_1
        }
    ]
}