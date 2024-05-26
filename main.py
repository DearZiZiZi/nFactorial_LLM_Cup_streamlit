import streamlit as st
from langchain.llms import OpenAI
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.callbacks.base import BaseCallbackHandler

import utils
import chatbot


# from dotenv import load_dotenv
# load_dotenv()
# import os
# API_KEY = os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title="Chatbot", page_icon="💬")
utils.configure_openai()

if "OPENAI_API_KEY" in st.session_state:
    vectorstore = chatbot.embeddings()



if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# st.session_state['OPENAI_API_KEY'] = API_KEY


st.header('Basic Chatbot')
st.write('Allows users to interact with the LLM')

class StreamHandler(BaseCallbackHandler):
    
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

class BasicChatbot:
    def __init__(self):
        self.openai_model = st.session_state['OPENAI_API_KEY']
    
    def setup_chain(self):
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)
        chain = ConversationChain(llm=llm, verbose=True)
        return chain

    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                # result = chain.invoke(
                #     {"input":user_query},
                #     {"callbacks": [st_cb]}
                # )
                # response = result["response"]
                response = chatbot.response(user_query, vectorstore)
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    obj = BasicChatbot()
    obj.main()
