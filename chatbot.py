# from dotenv import load_dotenv
# load_dotenv()
# import os
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.document_loaders import TextLoader , UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.faiss import FAISS
import pickle
from langchain_openai import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import streamlit as st




def embeddings():
    ## load excel file
    loader = UnstructuredExcelLoader(
        "./university_ranking.xlsx"
    )
    docs = loader.load()

    ## text splitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
    )

    documents = text_splitter.split_documents(docs)


    ## embeddings


    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state['OPENAI_API_KEY'])

    # Loading Vectors into VectorDB (FAISS)
    # As created by OpenAIEmbeddings vectors can now be stored in the database. The DB can be stored as .pkl file
    # 


    vectorstore = FAISS.from_documents(documents, embeddings)



    return vectorstore

# ## Chains
# With chain classes you can easily influence the behavior of the LLM



    # chain_type_kwargs = {"prompt": PROMPT}
    # llm = ChatOpenAI(openai_api_key=API_KEY)


## memory



## using memory in chains


def response(user_input, vectorstore):

    basePrompt = """
        Answer the questions based on the given context:
        {context}
        Question: {question}
        Answer here:
    """


    PROMPT = PromptTemplate(
        template=basePrompt, input_variables=["context", "question"]
    )


    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=st.session_state['OPENAI_MODEL'], temperature=0, openai_api_key=st.session_state['OPENAI_API_KEY'], verbose=True),
        memory=memory,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    # retriever=vectorstore.as_retriever()

    # print(retriever.invoke(user_input))

    response = qa.invoke({"question": user_input})
    # print("Sending Response...")
    # print(response)
    return response