import pandas as pd
import numpy as np
import pycaret
import pycaret.classification

#from pycaret.classification import *

def setup(data, target, use_gpu, outliar):
    return pycaret.classification.setup(data, target=target, session_id=123, train_size=0.7, use_gpu=use_gpu, remove_outliers=outliar)

def search_missing_value(data):
    import pandas as pd
    col = list(data.columns)
    missing_series = data.isnull().sum()
    missing_cols = []
    for i in col:
        if missing_series[i]!=0:
            missing_cols.append(i)
        else:
            missing_series = missing_series.drop(i)
    return missing_cols,missing_series

def interpolation(data,target,method): #target = missing_cols
    import pandas as pd
    for i in range(len(target)):
        data[target[i]] = data[target[i]].interpolate(method = method[i])
    return data

def save_df():
    results = pycaret.classification.pull()
    return results

def compare(standard):
    return pycaret.classification.compare_models(sort=standard)

def tune(model, opt):
    return pycaret.classification.tune_model(model, optimize=opt, choose_better=True)

def Blend(arr):
    arr[0]=pycaret.classification.create_model(arr[0])
    arr[1]=pycaret.classification.create_model(arr[1])
    arr[2]=pycaret.classification.create_model(arr[2])
    return pycaret.classification.blend_models([arr[0],arr[1],arr[2]])
def single(name):
    return pycaret.classification.create_model(name)

def single_visual(df):
    visual = df.iloc[0:9]
    return visual.plot()

def evaluate(model):
    return pycaret.classification.evaluate_model(model)

def prediction(model):
    return pycaret.classification.predict_model(model)

def save_model(model, name):
    return pycaret.classification.save_model(model, name)

def load(name):
    return pycaret.classification.load_model(name)