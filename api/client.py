##now over here in the client side of the things we will be making some new api routes to test the api

import requests 
import streamlit as st


##over here what this api is doing is that it is taking the input from the user and sending it to the api that we have made using fastapi and langserve
##see carefully that the key content and output will hold the respinse and input will hold the input to the api
def get_gemini_response(input_text):
    response=requests.post(
        "http://localhost:8000/essay/invoke",
        json={"input":{'topic': input_text}}
    )
    return response.json()['output']['content']


#streamlit framework
st.title("Langchain Client Side Application")
input_text=st.text_input("Search the topic u want")

##whenever i write anything in the text over there then it will call the get_gemini_response function
if input_text:
    st.write(get_gemini_response(input_text))