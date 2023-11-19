#import Streamlit App
import streamlit as st

#fix sqlite3 only on when run on Streamlit Cloud
#if "DEPLOYED_TO_CLOUD" in st.secrets and st.secrets["DEPLOYED_TO_CLOUD"] == 1:
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    print('info: did not require any changes to sqlite3 on this OS')

#imports
from langchain.chat_models import ChatCohere
from langchain.schema import HumanMessage
import cohere  
from langchain.llms import Cohere
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain #
from langchain.vectorstores import Qdrant #
from langchain.memory import ConversationBufferMemory
import os
from langchain.document_loaders import TextLoader
from chromadb.utils import embedding_functions
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
print(config['LANGCHAIN_COHERE']['MAX_TOKENS'])


#   Get Parameter Variables
if "temperature" not in st.session_state:
    st.session_state.temperature = float(config['LANGCHAIN_COHERE']['TEMPERATURE'])
    
if "search_k" not in st.session_state:
    st.session_state.search_k = int(config['LANGCHAIN_COHERE']['SEARCH_K'])

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = int(config['LANGCHAIN_COHERE']['CHUNK_SIZE'] )

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = int(config['LANGCHAIN_COHERE']['MAX_TOKENS'])

if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = float(config['LANGCHAIN_COHERE']['CHUNK_OVERLAP'] )

if "chat_input" not in st.session_state:
    st.session_state.chat_input = config['STREAMLIT_UI']['CHAT_INPUT'] 

if "init_assistant_message" not in st.session_state:
    st.session_state.init_assistant_message = config['STREAMLIT_UI']['INIT_ASSISTANCE_MESSAGE'] 

api_key=st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else None
if api_key == None:
    raise Exception('Require a COHERE_API_KEY to be setup in as an environment variable')    

st.title(config['STREAMLIT_UI']['TITLE'])

#@st.cache_resource
def setup_chain():
    #Initalize Cohere
    co = cohere.Client(api_key)
    #I put the text file into u_data folder, you can usde your path
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader("u_data/", silent_errors=False, show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    len(docs)
    embeddings = CohereEmbeddings(cohere_api_key=api_key)
    text_splitter = CharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
    documents = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents, embeddings) #db
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    model =ChatCohere(model="command-nightly",max_tokens=st.session_state.max_tokens,temperature=st.session_state.temperature, cohere_api_key=api_key)

    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=index.vectorstore.as_retriever(search_kwargs={"k": st.session_state.search_k}), verbose = True, chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        })
    
    return chain

if "chain" not in st.session_state:
    st.session_state.chain = setup_chain()

question = ""

#Initialize the streamlit App
def submit_parameters():
    st.session_state.chunk_size = int(st.session_state.chunk_size_parameter)
    st.session_state.search_k = int(st.session_state.search_k_parameter)
    st.session_state.temperature = float(st.session_state.temperature_parameter)
    st.session_state.max_tokens = int(st.session_state.max_tokens_parameter)
    st.session_state.chunk_overlap = float(st.session_state.chunk_overlap_parameter)
    st.session_state.messages = []
    del st.session_state["chain"]
    #st.cache_resource.clear()

with st.sidebar.form("Parameters"):
    st.slider('Temperature',0.00,1.00,st.session_state.temperature,.05, key="temperature_parameter")
    st.slider('Search K',1,10,st.session_state.search_k,1, key="search_k_parameter")
    st.slider('Chunk Size',200,2500,st.session_state.chunk_size,100, key="chunk_size_parameter")
    st.slider('Max Tokens',100,1000,st.session_state.max_tokens,100, key="max_tokens_parameter")
    st.slider('Chunk Overlap',0.00,0.50,st.session_state.chunk_overlap,0.05, key="chunk_overlap_parameter")
    submitted = st.form_submit_button("Submit and Reset", on_click=submit_parameters)
     
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input(st.session_state.chat_input):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

# First default response from Assistance
response = st.session_state.init_assistant_message

if question != None:
    #response = f"{list(chain(prompt).values())[1]}"
    response = st.session_state.chain.run({"query": question})
# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})

#query=""
#while query!="stop":
#    query = input("Prompt: ")
#    if query=="stop":
#        print("stoped by user")
#    print(list(chain(query).values())[1])