import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

#load the trained model
model=tf.keras.models.load_model('model.h5')

#load the scaler and encoder
with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)
with open('onehotencoder.pkl','rb') as f:
    ohe=pickle.load(f)
with open('labelencoder.pkl','rb') as f:
    le=pickle.load(f)

#streamlit app
st.title('Bank Customer Churn Prediction')

#input fields
geography=st.selectbox('Geography',ohe.categories_[0])
gender=st.selectbox('Gender',le.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score') 
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[le.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[0]  #dummy value
})

geo=ohe.transform([[geography]]).toarray()
geo_df=pd.DataFrame(geo,columns=ohe.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)

#scale the input data
input_data_scaled=scaler.transform(input_data)

#make prediction
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

#display the result
if prediction_prob>0.5:
    st.write(f'The customer is likely to churn with a probability of {prediction_prob:.2f}')
else:
    st.write(f'The customer is unlikely to churn with a probability of {1-prediction_prob:.2f}')