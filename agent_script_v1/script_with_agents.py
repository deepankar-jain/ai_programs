#!/usr/bin/env python
# coding: utf-8


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

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
# Set the API key for the OpenAI API client
# openai.api_key = api_key
from langchain.utilities import WikipediaAPIWrapper


# Now you can use the OpenAI API client with your loaded API key


def main():
    load_dotenv()
    st.set_page_config(page_title="script creator")
    st.title("GPT title and script creator")
    # st.title("test")
    prompt = st.text_input("What topic are you intersted in") 
    

    title_template = PromptTemplate( 
        input_variables = ['topic'], 
        template = 'Suggest me 1 great name for personal blogging websites for the following topic: {topic}'
    )

    script_template = PromptTemplate( 
        input_variables = ['title','wikipedia_search'], 
        template = 'Write me an introductory template on this title {title} while leveraging this wiki search {wikipedia_search}'
    )

    title_memory = ConversationBufferMemory(input_key = 'topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key = 'title', memory_key='chat_history')

    llm = OpenAI(model='text-curie-001', temperature=0.9)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=llm, prompt = script_template, verbose = True, output_key='script', memory=script_memory)


    wiki = WikipediaAPIWrapper()


    if prompt:
        with get_openai_callback() as cb:
            title=title_chain.run(prompt) # not sure on the variable here
            wiki_research = wiki.run(prompt)
            script= script_chain.run(title=title, wikipedia_search=wiki_research)
            print(cb)

        st.write(title)
        st.write(script)

        with st.expander('Title History'): 
            st.info(title_memory.buffer)

        with st.expander('Script History'): 
            st.info(script_memory.buffer)

        with st.expander('Wikipedia Research'): 
            st.info(wiki_research)
if __name__ =='__main__':
    main()