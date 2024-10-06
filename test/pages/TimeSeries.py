import streamlit as st
import AutoRegression as AR
import pandas as pd
import time
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt
from tqdm.notebook import tqdm
import tensorflow as tf
tf.random.set_seed(2)
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import itertools
import statsmodels.api as sm
import random
import seaborn as sns
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model,save_model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, Activation, Dense, Lambda, Concatenate, add, Flatten,SpatialDropout1D, LSTM
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from sklearn.utils import shuffle, class_weight

import itertools
st.set_page_config(layout="wide")
def print_static_rmse(actual, predicted, start_from=0,verbose=0):
    rmse = np.sqrt(mean_squared_error(actual[start_from:],predicted[start_from:]))
    std_dev = actual[start_from:].std()
    if verbose == 1:
        print('    RMSE = %0.2f' %rmse)
        print('    Std Deviation of Actuals = %0.2f' %(std_dev))
        print('    Normalized RMSE = %0.1f%%' %(rmse*100/std_dev))
    return rmse


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
        if method[i] in ['linear', 'pad','index']:
            data[target[i]] = data[target[i]].interpolate(method = method[i])
        else:
            data[target[i]] = data[target[i]].interpolate(method = method[i], order=3)
    return data

def apply_time_step(data_frame, ts, lag):
    data = data_frame.copy()

    X = []
    Y = []

    for i in range(len(data)-ts-(lag-1)):
        X.append(data.values[i:i+ts, :])
        Y.append(data.values[i+ts+(lag-1), :])

    X = np.array(X,dtype=np.float32)
    Y = np.array(Y,dtype=np.float32)

    return X, Y


def data_split(data_frame, ratio, ts, lag):
    train = ratio[0]
    validation = ratio[1]
    data = data_frame.copy()
    
    X, Y = apply_time_step(data, ts, lag)

    num_train = int(X.shape[0] * train)
    num_validation = int(X.shape[0] * (train+validation))
    
    X_train = X[:num_train, :, :]
    Y_train = Y[:num_train, ]

    X_val = X[num_train:num_validation, :, :]
    Y_val = Y[num_train:num_validation, ]

    X_test = X[num_validation:, :, :]
    Y_test = Y[num_validation:, ]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def ResBlock(x, Act, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(x) 
    r = BatchNormalization()(r) 
    r = Activation(Act)(r)
    r = SpatialDropout1D(0.3)(r)
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(r) 
    r = BatchNormalization()(r) 
    r = Activation(Act)(r)
    r = SpatialDropout1D(0.3)(r)
    if x.shape[-1]==filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x) #shortcut (shortcut)
    o = add([r,shortcut])

    return o
 

def TCN(Res_num, los, Act, X_train, Y_train, X_val, Y_val):
    inputs = Input(shape=(X_train.shape[1],X_train.shape[2]))
    z=0
    for i in range(Res_num,0,-1):
        if i==Res_num:
            z+=1
            x = ResBlock(inputs, Act, filters=32, kernel_size=2**i, dilation_rate=z)
        else:
            z+=1
            x = ResBlock(x, Act, filters=16,  kernel_size=2**i, dilation_rate=z)
    x = Lambda(lambda y: y[:,-1])(x)
    x = Flatten()(x)
    x = Dense(1)(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss=los)
    
    return model

def vis(history,name) :
    plt.title(f"{name.upper()}")
    plt.xlabel('epochs')
    plt.ylabel(f"{name.lower()}")
    value = history.history.get(name)
    val_value = history.history.get(f"val_{name}",None)
    epochs = range(1, len(value)+1)
    plt.plot(epochs, value, 'b-', label=f'training {name}')
    if val_value is not None :
        plt.plot(epochs, val_value, 'r:', label=f'validation {name}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.05, 1.2) , fontsize=10 , ncol=1)
    
def plot_history(history) :
    key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
    fig = plt.figure(figsize=(12, 4))
    for idx , key in enumerate(key_value) :
        plt.subplot(1, len(key_value), idx+1)
        vis(history, key)
    plt.tight_layout()
    return fig



st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("# Time Series Analysis(One to One)")
st.sidebar.markdown("# Time Series Analysis")

if "TimeSeries_disabled" not in st.session_state:
    st.session_state.TimeSeries_disabled = True

r_data_=None

def uploader_callback():
    if st.session_state.TimeSeries_disabled== False:
        st.session_state.TimeSeries_disabled = True

r_data_ = st.file_uploader(
    label='Time-Series file upload',
    on_change=uploader_callback,
    key='Time-Series file_uploader'
)

method_box=[]
analysis_data = None
check_model = False
setup_ready = False


if r_data_ is not None:
    r_data = pd.read_csv(r_data_)
    st.subheader('Original Data Head')
    with st.expander("Start"):
        st.write(r_data.head())
        if st.checkbox('Show Data Describe'):
            st.subheader('Data Describe')
            if analysis_data is not None:
                st.write(analysis_data.describe())
            else:
                st.write(r_data.describe())

    st.subheader('Data Visiaulize')
    with st.expander("Start"):      
        te_data = r_data.copy()
        option = st.selectbox(
            'Select Variable', 
            (te_data.columns),
            )
        show = st.button("Show")
        if show:
            st.line_chart(te_data[option])
        
    st.subheader('Data Pre-processing')
    with st.expander("Start"):
        missing_cols,missing_series = search_missing_value(r_data)
        if len(missing_cols) != 0 and len(missing_series) != 0:
            missing_check = 1
        else:
            missing_check = 0
        st.subheader('Missing Value')
        if missing_check == 1:
            st.write(missing_cols)
            st.write(missing_series)
            for i in missing_cols:
                method = st.selectbox(f"Variable: {i}",["linear", "index","pad","nearest"],key=i)
                method_box.append(method)
        else:
            st.write("Not Found Missing Value")
        
        analysis_data = interpolation(r_data, missing_cols, method_box) 
        
        target_ = st.selectbox('분석을 원하는 대상을 선택해 주세요.', list(analysis_data.columns), len(analysis_data.columns)-1)
        base_index = st.selectbox('시간을 나타내는 변수를 선택해주세요. 해당 변수를 기준으로 정렬됩니다.', list(analysis_data.columns), 0)
        train = st.slider('Train Data 비율', 0, 100, 60, 10)
        val = st.slider('validation Data 비율', 0, 100, 20, 10)
        test = st.slider('Test Data 비율', 0, 100, 20, 10)
        train = train/100
        val = val/100
        test = test/100
        ts = st.number_input("Time step : 시퀀스의 길이, 즉 모델에 대한 입력으로 사용되는 과거 관찰 수")
        ts = int(ts)
        lag = st.number_input("Lag : TS와 예측값의 시간 지연량")
        lag = int(lag)
        base_model = st.selectbox('분석에 사용할 Model을 선정해주세요.', ['Temporal Convolutional Networks', 'Long Short Term Memory'], 0)

        

        analysis_data = analysis_data.sort_values(base_index)
        analysis_data = analysis_data[target_]
        analysis_data=np.array(analysis_data)
        train, validation, test = data_split(pd.DataFrame(analysis_data),[train,val,test], ts, lag)

        x_train, y_train = train
        x_val, y_val = validation
        x_test, y_test = test

        
        if st.checkbox('Show train, validation, test data shape'):
            st.subheader('data_shape')
            if analysis_data is not None:
                st.write('X_train : {}, y_train : {}'.format(x_train.shape,y_train.shape))
                st.write('X_val : {}, y_val : {}'.format(x_val.shape, y_val.shape))
                st.write('X_test : {}, y_test : {}'.format(x_test.shape, y_test.shape))
            else:
                st.write("Split을 먼저 진행해주세요.")

        btn_clicked = st.button("Confirm", key='confirm_btn')

        if btn_clicked:
            con = st.container()
            con.caption("Ready!")
            st.session_state.TimeSeries_disabled = False
try:
    if analysis_data is not None:
        st.subheader('%s Setup'%base_model)
        if base_model=='Temporal Convolutional Networks':
            with st.expander("Start"):
                Act = st.selectbox('활성화 함수를 선택해주세요.', ['relu', 'sigmoid', 'tanh'], 0)
                loss = st.radio(label = 'loss function을 선택해주세요.', options = ['mae', 'mse'])
                st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) 
                Batch = st.number_input("Batch Size를 설정해주세요.")
                Batch = int(Batch)
                epochs = st.number_input("Epochs를 설정해주세요.")
                epochs = int(epochs)
                Res_num = st.number_input("Residual Block의 개수를 설정해주세요. 1개의 Residual Block은 Conv1D, BatchNormalization, Activation, Drop_Out이 2번 반복되어 총 10개의 Layer로 구성됩니다.")
                Res_num = int(Res_num)
                confirm = st.button("confirm", disabled=st.session_state.TimeSeries_disabled, key="confirm_btn2")
                if confirm:
                    check_model = True
                if check_model==True:
                    progress_bar = st.progress(0)
                    model = TCN(Res_num, loss, Act, x_train, y_train, x_val, y_val)
                    model.save('./pages/models/base_model.h5')
                    for percent_complete in range(100):
                        time.sleep(0.1)  
                        progress_bar.progress(percent_complete + 1)
                    con = st.container()
                    con.caption("Model Compile!")
                    st.subheader('Model Summary')
                    model.summary(print_fn=lambda x: st.text(x))
                    con = st.container()
                    con.caption("base model은 ./pages/models/base_model.h5 형태로 저장됩니다! ")
                    html = """
                            아래의 링크에서 저장된 h5파일을 업로드하면 모델의 구조를 확인할 수 있습니다. <br/><b> [netron](https://netron.app/)</b>
                        """
                    st.markdown(html, unsafe_allow_html=True)

        if base_model=='Long Short Term Memory':
            with st.expander("Start"):
                Act = st.selectbox('활성화 함수를 선택해주세요.', ['relu', 'sigmoid', 'tanh'], 0)
                los = st.radio(label = 'loss function을 선택해주세요.', options = ['mae', 'mse'])
                st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) 
                Batch = st.number_input("Batch Size를 설정해주세요.")
                Batch = int(Batch)
                epochs = st.number_input("Epochs를 설정해주세요.")
                epochs = int(epochs)
                inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
                x = LSTM(units=100, return_sequences=True, activation=Act)(inputs)
                x = Dropout(0.2)(x)
                x = LSTM(units=100, return_sequences=False, activation=Act)(x)
                x = Dropout(0.2)(x)
                outputs = Dense(1)(x)
                model = Model(inputs=inputs, outputs=outputs)
                confirm = st.button("confirm", disabled=st.session_state.TimeSeries_disabled, key="confirm_btn2")
                if confirm:
                    check_model = True
                if check_model==True:
                    progress_bar = st.progress(0)
                    model.compile(optimizer='adam', loss=los)
                    model.save('./pages/models/base_model.h5')
                    for percent_complete in range(100):
                        time.sleep(0.1)  
                        progress_bar.progress(percent_complete + 1)
                    con = st.container()
                    con.caption("Model Compile!")
                    st.subheader('Model Summary')
                    model.summary(print_fn=lambda x: st.text(x))
                    con = st.container()
                    con.caption("base model은 ./pages/models/base_model.h5 형태로 저장됩니다! ")
                    html = """
                            아래의 링크에서 저장된 h5파일을 업로드하면 모델의 구조를 확인할 수 있습니다. <br/><b> [netron](https://netron.app/)</b>
                        """
                    st.markdown(html, unsafe_allow_html=True)

        st.subheader('Train')
        with st.expander("Start"): 
            model.optimizer.lr = 0.001
            mc = ModelCheckpoint('./pages/models/Train_model.h5', monitor='val_loss', mode='min', save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='min')

            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
            with st.spinner("Train...."):
                his = model.fit(x_train, y_train,
                                batch_size=Batch, epochs=epochs, verbose=1,
                                validation_data=(x_val, y_val), callbacks=[early_stopping, mc])
            st.write('Train loss : ',his.history['loss'])
            st.write('Validation loss : ',his.history['val_loss'])
            st.subheader("Train vs Val")
            fig = plot_history(his)
            st.pyplot(fig)

        st.subheader('Test')
        with st.expander("Start"): 
            with st.spinner("Predict"):
                pre=model.predict(x_test)
            rmse= print_static_rmse(y_test, pre, 0, 0)
            r2_y_predict = r2_score(y_test, pre)
            st.write('RMSE : ',rmse)
            st.write('R-squared : ',r2_y_predict)
            st.subheader("Time Series Graph")
            fig2 = plt.figure(figsize=(16,8))
            plt.plot(pre,label='Predict')
            plt.plot(y_test,label='Observation')
            plt.ylabel('%s'%target_)
            plt.xlabel('Time')
            plt.legend()
            st.pyplot(fig2)
            con = st.container()
            con.caption("Train model은 ./pages/models/Train_model.h5 형태로 저장됩니다! ")
except:
    st.write("단계 별로 진행해주세요.")
            



            
