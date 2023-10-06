# Этот модуль еще не готов

import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from YaGPT import YaGPTEmbeddings, YandexLLM

st.set_page_config(page_title="Чат с документами Streamlit на базе LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
#openai.api_key = st.secrets.openai_key
folder_id = st.secrets.folder_id
yagpt_key = st.secrets.yagpt_key
instructions = """
    Вы являетесь экспертом по библиотеке Streamlit Python, и ваша задача - отвечать на технические вопросы. Предполагается, что все вопросы связаны с библиотекой Streamlit Python. Ответы должны быть техническими и основанными на фактах - не фанатазируй о возможностях."""
LLM = YandexLLM(api_key=api_key, folder_id=folder_id, instruction_text = instructions, temperature = 0.01)
st.title("Чат с документами Streamlit на базе LlamaIndex 💬🦙")
st.info("Ознакомьтесь с полным руководством по созданию этого приложения в нашем [блог-посте].(https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Задайте мне вопрос о Python-библиотеке Streamlit с открытым исходным кодом!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Загрузка и индексация документации Streamlit - держитесь крепче! Это займет 1-2 минуты."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        #service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
        service_context = ServiceContext.from_defaults(llm=LLM)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features.")

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features. Respond in Russian")

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
