## so now we will be making an api using the model gemini flash 1.5 using fastapi and langserve

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate  # Fixed import
##importing the gemini flash 1.5 model from langchain
from langchain_google_genai import ChatGoogleGenerativeAI  # Fixed import
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Fixed environment variable name


##so here we will be making a fastapi app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Langchain Server for Gemini Flash 1.5"
)


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  


add_routes(app,
           model,
           path="/gemini"
           )

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Write an essay on the topic {topic} in 100 words")
])  

##now making up a route that will take the topic as input and return the essay
add_routes(
    app,
    prompt1 | model,  # Fixed chain order
    path="/essay"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)