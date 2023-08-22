
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, OpenAI, LLMChain
import openai 
import os
from dotenv import load_dotenv
import streamlit as st
# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
import time
# Load the chain
import pickle


# Set up your OpenAI API key
#openai.api_key = "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"


# Load environment variables from .env file
load_dotenv()

# Fetch the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")





import os

@st.cache_resource
def libraries():

    

    llm=OpenAI(temperature=1)

    #llm = OpenAI(temperature=0,)
    # Load the question answering chain
    c = load_qa_chain(llm, chain_type="stuff")

    embeddings = HuggingFaceEmbeddings()


    return c 

c = libraries()

# Text Splitter





# Load the chain
@st.cache_resource
def model():

    load_path = "db.pkl"
    with open(load_path, "rb") as af:
        d = pickle.load(af)
        return d
d  = model()


def main():
    st.title("TESTING")

    # User input
    input_text = st.text_input("Enter the starting sentence")

    # Generate story when button is clicked
    if st.button("ENTER"):
        progress_text = "Generating answer...🤗"
        my_bar = st.progress(0)
        my_bar_text = st.empty()

        for percent_complete in range(100):
            time.sleep(.01)
            my_bar.progress(percent_complete + 1)
            my_bar_text.text(f"{progress_text} {percent_complete + 1}%")
        docs = d.similarity_search(input_text)
        a = c.run(input_documents=docs, question=input_text) 
        generated_story = a

        # Display the generated story
        #st.write("Generated Story:")
        st.text_area("Generated anwer", generated_story, height=200)

if __name__ == "__main__":
    main()
