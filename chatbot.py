#not all imports really needed, I have to clean
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
import streamlit as st
from dotenv import load_dotenv


@st.cache_resource
def setup_chain():
    #Get Environment Variables
    load_dotenv()
    api_key=os.getenv('COHERE_API_KEY')
    if api_key == None:
        raise Exception('Require a COHERE_API_KEY to be setup in as an environment variable')
    max_tokens=int(os.getenv('MAX_TOKENS') or 600)
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = float(os.getenv('TEMPERATURE') or 0.75)
    
    if "search_k" not in st.session_state:
        st.session_state.search_k = int(os.getenv('SEARCH_K') or 1)

    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = int(os.getenv('CHUNK_SIZE') or 1000)

    print('max_tokens',max_tokens,'temperature',st.session_state.temperature,'search_k',st.session_state.search_k, 'chunk_size', st.session_state.chunk_size)
    
    #Initalize Cohere
    co = cohere.Client(api_key)
    #I put the text file into u_data folder, you can usde your path
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader("u_data/", silent_errors=False, show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    len(docs)
    embeddings = CohereEmbeddings(cohere_api_key=api_key)
    text_splitter = CharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents, embeddings) #db
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    model =ChatCohere(model="command-nightly",max_tokens=max_tokens,temperature=st.session_state.temperature, cohere_api_key=api_key)

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

chain = setup_chain()

question = ""

#Initialize the streamlit App
def submit_parameters():
    st.session_state.chunk_size = int(st.session_state.chunk_size_parameter)
    st.session_state.search_k = int(st.session_state.search_k_parameter)
    st.session_state.temperature = float(st.session_state.temperature_parameter)
    st.session_state.messages = []
    st.cache_resource.clear()

with st.sidebar.form("Parameters"):
    st.slider('Temperature',0.00,1.00,st.session_state.temperature,.05, key="temperature_parameter")
    st.slider('Search K',1,10,st.session_state.search_k,1, key="search_k_parameter")
    st.slider('Chunk Size',200,2500,st.session_state.chunk_size,100, key="chunk_size_parameter")
    submitted = st.form_submit_button("Submit", on_click=submit_parameters)
     


st.title("Program Selection")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input("What do you want to know about?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

# First default response from Assistance
response = "I am the U of T Program Selection Assistant - How may I help you?"

if question != None:
    #response = f"{list(chain(prompt).values())[1]}"
    response = chain.run({"query": question})
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