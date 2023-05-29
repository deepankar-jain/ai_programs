#!/usr/bin/env python
# coding: utf-8

# In[18]:


# !pip install langchain
# !pip install pypdf2 streamlit
import openai
import os
import json
import requests
from dotenv import load_dotenv
# Load environment variables from .env file
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for the OpenAI API client
# openai.api_key = api_key


# Now you can use the OpenAI API client with your loaded API key


# In[20]:


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask PDF")
    st.header("Ask your pdf")
    
    pdf = st.file_uploader("Upload you pdf", type="pdf")
    
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
#             print(page)
            text+=page.extract_text()
#         st.write(text)

    #Split into chunks
        text_splitter= CharacterTextSplitter(separator="\n",
        chunk_size=1000, chunk_overlap=200,length_function=len
        )
        chunks=text_splitter.split_text(text)
#         st.write(chunks)
    #Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
    #Ask questions
        user_question = st.text_input("Ask a question to your PDF")
        
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
#             st.write(docs)
#             llm = OpenAI(model="text-ada-001") #cheapest but not good
#             llm = OpenAI(model="text-davinci-002") #  expensive
            llm = OpenAI(model="text-curie-001") # cheap and okay
#             llm = OpenAI(model="text-babbage-001") #cheaper but not good
    
            chain = load_qa_chain(llm, chain_type = "stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents= docs, question= user_question)
                print(cb)
            
            st.write(response)
#         st.write(knowledge_base)
if __name__ =='__main__':
    main()


