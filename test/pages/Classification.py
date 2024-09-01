import streamlit as st
import AutoClassification as AC
import pandas as pd
from streamlit_shap import st_shap
import shap as sp

st.markdown("# Classification")
st.sidebar.markdown("# Classification")

c_data_ = st.file_uploader('upload classification data')

classification_flag = st.button("start")

if c_data_:
    c_data = pd.read_csv(c_data_)

    missing_cols,missing_series = AC.search_missing_value(c_data)
    if len(missing_cols) != 0 and len(missing_series) != 0:
        c_data = AC.interpolation(c_data, missing_cols,
                                ['linear','linear'])  
        missing_check = 1
    else:
        missing_check = 0

    st.subheader('Missing Value')
    if missing_check == 1:
        st.write(missing_cols)
        st.write(missing_series)
    else:
        st.write("Not Found Missing Value")

    st.subheader('Data header')
    st.write(c_data.head())

    # 시각화를 할지 말지...
    st.subheader('Data Visiaulize')
    st.write("...")

    st.subheader('Setting Model')
    target_ = st.selectbox('Select Target', c_data.columns, len(c_data.columns)-1)
    setting = AC.setup(c_data, target_, True, True)
    setting_result = AC.save_df()

    metrics_arr = setting.get_metrics()
    metrics_arr = metrics_arr['Name']
    error_ = st.selectbox('Select Error', metrics_arr)


if classification_flag:
    st.subheader('Data Setup')
    with st.spinner("setting..."):
        st.write(setting_result)

    st.subheader('Model List')
    with st.spinner("model list..."):
        model_ = setting.models()
        st.write(model_)
        best_model = model_.index[0]
        st.write('Best Model :', best_model)
        
    st.subheader('Single Model')
    with st.spinner("single model..."):
        single_model = AC.single(best_model)
        single_result = AC.save_df()
        st.write(single_result)

    with st.spinner("single visual graph data"):
        single_visual_result = AC.single_visual(single_result)
        single_visual_graph = AC.save_df()
        st.write('Single Visual Graph Data')
        st.line_chart(single_visual_graph)
    

    st.subheader('Compare Model')
    with st.spinner("compare model.."):
        best_model = AC.compare(error_)
        compare_matrix = AC.save_df()
        st.write(compare_matrix)
    
    st.subheader('Model Tuning')
    with st.spinner('tuning...'):  
        optimize_best_model = AC.tune(best_model, error_)
        optimize_model_matrix = AC.save_df()
        st.write(optimize_model_matrix)

    #st.subheader("Ensemble based Soft Voting")
    #st.write("Choose 3-Model")

    st.subheader("Visual & Evaluate")
    with st.spinner("evaluate..."):
        AC.evaluate(optimize_best_model)
    
    st.write("Predict")
    with st.spinner("predict..."):
        pred = AC.prediction(optimize_best_model)
        st.write(pred)

    st.write("Save")
    with st.spinner("save model"):
        save_model = AC.save_model(optimize_best_model, 'pipeline')
        st.write(save_model)

    st.write("Model Load")
    with st.spinner("load model"):
        load = AC.load('pipeline')
        st.write(load)
        model_pred = AC.prediction(load)
        st.write(model_pred)

elif not c_data_ and classification_flag:
    st.error('데이터를 넣어주세요')