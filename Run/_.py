def pre(data,target,use_gpu,outliar):
    import pandas as pd
    import numpy as np
    import pycaret.regression 
    return pycaret.regression.setup(data, target = target, session_id = 123, train_size=0.7, 
                                    use_gpu=use_gpu, remove_outliers = outliar)
def save_df():
    import pycaret.regression 
    results = pycaret.regression.pull()
    return results

def compare(standard):
    import pycaret.regression 
    return pycaret.regression.compare_models(sort = standard,)

def tune(model,opt):
    import pycaret.regression
    return pycaret.regression.tune_model(model,optimize = opt, choose_better = True)

def single(name):
    import pycaret.regression
    return pycaret.regression.create_model(name)

def single_visual(df):
    import pandas as pd
    visual = df.iloc[0:9]
    return visual.plot()

def evaluate(model):
    import pycaret.regression
    #return pycaret.regression.plot_model(model, plot = shape)
    return pycaret.regression.evaluate_model(model)

def shap(model):
    import pycaret.regression
    return pycaret.regression.interpret_model(model)

def prediction(model):
    import pycaret.regression
    return pycaret.regression.predict_model(model)

def save(model,name):
    import pycaret.regression
    return pycaret.regression.save_model(model,name)

def load(name):
    import pycaret.regression
    return pycaret.regression.load_model(name)

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

import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_csv('./dataset/testdata.csv')
print(df.head())
missing_cols,missing_series = search_missing_value(df)
if len(missing_cols) != 0 and len(missing_series) != 0:
    df = interpolation(df,missing_cols,
                                ['linear','linear'])  
    missing_check = 1
else:
    missing_check = 0


b = pre(df, 'Process', True, 'outliar')
print(b.models())
model_list = b.models()

model_list = model_list['Name']
print(model_list)



metrics_arr = b.get_metrics()
print(metrics_arr)
metrics_arr = metrics_arr['Name']
print(metrics_arr)


#
all_result = compare('MSE')
print(all_result)
save_df_result = save_df()
print(save_df_result)


    
#col1, col2, col3 = st.columns(3)
#with col1:
#    st.write(' ')
#with col2:
#    st.image("tuk.png")
#with col3:
#    st.write(' ')