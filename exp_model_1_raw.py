import numpy as np

#-----------------------------------------------------------------------
# ROOT DECOMPOSITION FUNCTIONS

def split_f_root(X):
    X_split = {0:  X[0][:-72,0:1],              # past-load 72-180 h before
               1:  X[0][-72:-24,0:1],           # past-load 24-72 h before
               2:  X[0][-24:-12,0:1],           # past-load 24-12 h before 
               3:  X[0][-12:,0:1],              # past-load 1-12 h before
               4:  X[0][:-24,1:4],              # past-calendar 24-180 h before’
               5:  X[0][-24:,1:4],              # past-calendar 1-24 h before’
               6:  X[1][:,0:3],                 # future-calendar’
               7:  X[0][:,4:5],                 # past-weather
               8:  X[1][:,3:4],                 # future_weather
               9:  [X[0][:,5:], X[1][:,4:]]     # holiday
    }         
    return X_split

def joint_f_root(X_split):
    X_past_load = np.concatenate(
        [X_split[0], X_split[1], X_split[2], X_split[3]], 
        axis = 0
    ).reshape(180, 1)   
    X_past_calendar = np.concatenate(
        [X_split[4], X_split[5]], 
        axis = 0
    ).reshape(180, 3)
    X_encoder = np.concatenate(
        [X_past_load, X_past_calendar, X_split[7], X_split[9][0]], 
        axis = 1
    ).reshape(180, 15)
    X_decoder = np.concatenate(
        [X_split[6], X_split[8], X_split[9][1]], 
        axis = 1
    ).reshape(36, 14)
    return [X_encoder,X_decoder]


explain_dict ={
    "name": "Seq2seq LSTM model RAW Interpretable Components",
    "component_names": [
        "past-load 72-180 h before",
        "past-load 24-72 h before",
        "past-load 12-24 h before",
        "past-load 1-12 h before",
        "past-calendar 24-180 h before",
        "past-calendar 1-24 h before",
        "future-calendar",                
        "past-weather", 
        "future-weather", 
        "holiday", 
    ],
    "joint_f": joint_f_root,
    "split_f": split_f_root,
}

