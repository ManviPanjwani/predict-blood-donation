#!/usr/bin/env python
# coding: utf-8

# In[44]:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image



st.markdown("<h1 style='text-align: center; color: #d7385e ;'><strong><u>Predict Blood Donation for Future Expectancy</u></strong></h1>", unsafe_allow_html=True)

image = Image.open('donate.jpeg')
st.image(image,width=700)

st.sidebar.markdown("<h1 style='text-align: center; color:#d7385e ;'><strong><u>Specify Input Parameters</u></strong></h1>", unsafe_allow_html=True)
    
st.markdown("Forecasting blood supply is a serious and recurrent problem for blood collection managers: in January 2019, Nationwide, the Red Cross saw 27,000 fewer blood donations over the holidays than they see at other times of the year. Machine learning can be used to learn the patterns in the data to help to predict future blood donations and therefore save more lives.")
st.markdown("Understanding the Parameters -")
st.markdown("(Recency - months since the last donation)")
st.markdown("(Frequency - total number of donations)")
st.markdown("(Monetary - total blood donated in c.c.)")
st.markdown("(Time - months since the first donation)")
st.markdown("Target - (1 stands for donating blood, 0 stands for not donating blood)")




def user_input_features():
    Recency  = st.sidebar.slider('Recency', 0, 74)
    Frequency= st.sidebar.slider('Frequency', 1,43)
    Monetary = st.sidebar.slider('Monetary', 250,12500)
    Time = st.sidebar.slider('Time', 2,98)

     
    data = {'Recency (months)': Recency  ,
           'Frequency (times)': Frequency,
           'Monetary (c.c. blood)': Monetary,
           'Time (months)':Time}
           
           
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.write(df)

transfusion = pd.read_csv('transfusion.zip')

transfusion.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True)

transfusion.target.value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.30,
    random_state=40,
    stratify=transfusion.target)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

prediction = logreg.predict(df)
prediction_proba = logreg.predict_proba(df)


st.subheader('Prediction')
st.write(transfusion.target[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)




