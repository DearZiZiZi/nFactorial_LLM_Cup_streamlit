import streamlit as st
from langchain.llms import OpenAI
import utils

from dotenv import load_dotenv
load_dotenv()
import os
API_KEY = os.environ.get("OPENAI_API_KEY")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# st.title('🦜🔗 AI Assistant chatbot')
# openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# def generate_response(input_text):
#     llm = OpenAI(temperature=0.7, openai_api_key=API_KEY)
#     st.info(llm(input_text))

# with st.form('my_form'):
#     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#     submitted = st.form_submit_button('Submit')
#     #if not openai_api_key.startswith('sk-'):
#     #    st.warning('Please enter your OpenAI API key!', icon='⚠️')
#     if submitted: # and openai_api_key.startswith('sk-'):
#         generate_response(text)

    #    'sk-proj-qkjqnjj5Y33tVUqdG7myT3BlbkFJbguWoZHf6k1NvQYZ39Lx'


import utils
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.callbacks.base import BaseCallbackHandler

st.session_state['OPENAI_API_KEY'] = API_KEY

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Anuar AI')
st.write('Intelligent platform designed to help students find suitable universities based on their specific criteria')
#st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/1_%F0%9F%92%AC_basic_chatbot.py)')

class StreamHandler(BaseCallbackHandler):
    
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

class BasicChatbot:
    def __init__(self):
        self.openai_model = "gpt-4o"
    
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
                result = chain.invoke(
                    {"input":user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["response"]
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = BasicChatbot()
    obj.main()
