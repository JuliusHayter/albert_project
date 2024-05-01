import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from datetime import datetime
import pickle



st.sidebar.image('/Users/julius/Downloads/airbnb_logo_icon_170605.png', width=150)
st.sidebar.write("## Would you like to rent out your apartment on AirBnB? \n This estimator calculates the ideal market price for your apartment based on its characteristics. ")

st.sidebar.write("Our prediction model has been trained on AirBnB offer data. Some features, which are difficult to evaluate by our model, are not taken into account (such as brightness or furniture quality). It is therefore important that you use this tool as a basis for your estimate, as a correction may be added if you consider your property to be exceptional.")

st.sidebar.write("This tool is brought to you by Paul Dominique, Promesse, Cristian, Bastien and Julius.")
st.sidebar.write("This content is free, please help us ! ")
st.sidebar.text_input('Your credit card number : ', max_chars=16)
st.sidebar.date_input("Expire", min_value=datetime(1900, 5,1), max_value=datetime(2100, 12,31), format='DD/MM/YYYY')
st.sidebar.text_input('Security code : ',max_chars=3)
st.sidebar.write('We guarantee not to charge more than 1000€ (per month)')


st.image("/Users/julius/Desktop/Airbnb_Logo_Bélo.svg.png", clamp=True)
for i in range(3):st.write("")
st.subheader("    Estimate the price at which you can rent your apartment ")
for i in range(3):st.write("")

size = st.number_input("Size of your appartment (square meters)", 9,1000)
for i in range(3):st.write("")
capacity = st.number_input("Capacity (the maximum number of travellers that can be accommodated in your apartment)", 1,10000)
for i in range(3):st.write("")
nb_of_bedrooms = st.number_input("Number of bedrooms", 1,10000)
for i in range(3):st.write("")
nb_of_beds = st.number_input("Number of beds",1,4)
for i in range(3):st.write("")
nb_of_bathrooms =  st.number_input("Number of bathrooms",1,4)
for i in range(3):st.write("")
arrondissement = st.number_input("Paris arrondissement in which your apartment is located ",1,20)
for i in range(3):st.write("")

test = np.concatenate((np.array([1, 2, 3, 4, 5, 6]), np.zeros(350)))
with open('/Users/julius/Desktop/école/python/ML/ML_POC/albert_project/notebooks/model.pkl', 'rb') as f:
    model = pickle.load(f)



if st.button("Get your estimation"):
    for i in range(3):st.write("")
    st.write("Our price estimation for your appartment is",int(model.predict(np.array([size, capacity, nb_of_bedrooms, nb_of_beds, nb_of_bathrooms, arrondissement]).reshape(1,-1))),"€ per night ! ")
    #st.success("Data loaded successfully!")
    st.balloons()
    st.camera_input("")

    st.image("https://www.cravate-avenue.com/modules/prestablog/views/img/grid-for-1-7/up-img/thumb_365.jpg?5a67ff59f28244882f504b0e4456010a", width=300,clamp=True)


for i in range(3):st.write("")

st.download_button(
    label="Download the pkl file to get our ML model",
    data="/Users/julius/Desktop/école/python/ML/ML_POC/albert_project/notebooks/model.pkl",
    file_name='model.pkl',
)












# # Generating a fake dataset
# np.random.seed(42)  # For reproducible results
# x = np.arange(10)  # X values
# y = np.random.rand(10) * 100  # Y values, randomly generated

# # Creating a plot
# st.subheader("Fake Data Visualization")
# fig, ax = plt.subplots()
# ax.bar(x, y, color='b', label="Fake Data")
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_title("Simple Plot of Fake Data")
# ax.legend()

# # Display the plot in Streamlit
# st.pyplot(fig)

# # Display a dataframe in Streamlit
# st.subheader("Quotes Data")

# #url to paste : https://quotes.toscrape.com/
# base_url = st.text_input("Enter the URL to scrape:")
# if base_url:
#     quotes_data = fetch_quotes(base_url)
#     st.write(pd.DataFrame(quotes_data))
#     st.success("Data loaded successfully!")