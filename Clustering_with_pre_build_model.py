#!/usr/bin/env python
# coding: utf-8

# # Moodel Deployment

# In[ ]:


import pandas as pd
import streamlit as st 
from sklearn.svm import SVC
from pickle import dump
from pickle import load

st.title('Model Deployment: Customer Personality Analysis')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Education = st.sidebar.selectbox('Select your education level', ['Basic','Graduation','Masters','phD','2n Cycle'], index=0)
    Marital_Status = st.sidebar.selectbox('Select Marital Status', ['Single', 'Married', 'Together', 'Divorced', 'Widow', 'Alone'], index=0)
    Income = st.sidebar.number_input('Income per Annual')
    Kidhome = st.sidebar.number_input("Kids in Home")
    Teenhome = st.sidebar.number_input("Teen in Home")
    Recency = st.sidebar.number_input("Recency")
    MntWines = st.sidebar.number_input("MntWines")
    MntFruits = st.sidebar.number_input("MntFruits")
    MntMeatProducts = st.sidebar.number_input("MntMeatProducts")
    MntFishProducts = st.sidebar.number_input("MntFishProducts")
    MntSweetProducts = st.sidebar.number_input("MntSweetProducts")
    MntGoldProds = st.sidebar.number_input("MntGoldProds")
    data = {'Education':Education,
            'Marital_Status':Marital_Status,
            'Income':Income,
            'Kidhome':Kidhome,
            'Teenhome':Teenhome,
            'Recency': Recency,
            'MntWines' : MntWines,
            'MntFruits' : MntFruits,
            'MntMeatProducts' : MntMeatProducts,
            'MntFishProducts' : MntFishProducts,
            'MntSweetProducts' : MntSweetProducts,
            'MntGoldProds': MntGoldProds}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

# load the model from disk
loaded_model = load(open(r"final_model.pkl", 'rb'))

df = pd.get_dummies(df, columns=['Education', 'Marital_Status'])

prediction = loaded_model.predict(df)

st.subheader('Predicted Result')

st.subheader('Customer Prediction ')
st.write(prediction)

