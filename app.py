import os
import time
import streamlit as st
import tempfile
from dotenv import load_dotenv


##Langchain core imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import  ChatPromptTemplate, MessagesPlaceholder


##Langchain LLMs and Chains
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever,create_retrieval_chain


##Text splitting and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


##vector store (FAISS)
from langchain.vectorstores import FAISS

##PDF Loader
from langchain_community.document_loaders import PyPDFLoader


##load environment variables
load_dotenv()


##set page config
st.set_page_config(
    page_title="📄 RAG Q&A Chatbot",
    layout="wide",
    initial_sidebar_state = "expanded"
)


st.title("📄 Retrieval-Augmented Q&A with Backend PDFs + Chat History")

st.sidebar.header("⚙️ Configuration")
st.sidebar.write(
    "- Enter your GROQ API key below\n"
    "- Backend PDFs will auto-load\n"
    "- Ask questions and see full chat history!"
)

##API key input
api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")
if not api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()



##embeddings setup
st.sidebar.subheader("Embedding Model")
st.sidebar.write("Using: `sentence-transformers/all-MiniLM-L6-v2`")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


DATA_DIR ="."
SELECTED_PDFS = ["Docker.pdf"]




@st.cache_resource(show_spinner = True)
def load_pdfs_from_backend():
    all_docs = []
    for pdf_path in SELECTED_PDFS:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs
with st.spinner("📚 Loading backend PDFs..."):
    all_docs = load_pdfs_from_backend()

if not all_docs:
    st.error("❌ No PDFs found in `data/` folder. Please add at least one PDF.")
    st.stop()
st.success("✅ PDFs loaded successfully from backend!")

st.sidebar.subheader("📄 Loaded PDFs:")
for filename in SELECTED_PDFS:
    st.sidebar.write(f"- {os.path.basename(filename)}")



        ##split Text into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap = 150,
)
splits = text_splitter.split_documents(all_docs)


##create vector store

@st.cache_resource(show_spinner=False)
def get_vectorstore(_splits):
    return FAISS.from_documents(_splits, embeddings)

vectorstore = get_vectorstore(splits)
retriever = vectorstore.as_retriever()

##setup groq LLM
llm = ChatGroq(groq_api_key = api_key, model_name="llama-3.1-8b-instant",temperature=0  )

##RAG Components
contextualize_q_prompt = ChatPromptTemplate.from_messages([             ##“Chat history aur user ke naye question ko dekh kar decide karo ki humein database se kya retrieve karna chahiye.”
    ("system", "Given the chat history and the latest user question, decide what to retrieve."),
    MessagesPlaceholder("chat_history"),   ##MessagesPlaceholder("chat_history") → yahan purani conversation insert hoti hai.
    ("human", "{input}"),  ##"{input}" → ye current user ka latest question hota hai.
])

history_aware_retriever = create_history_aware_retriever(      ###create_history_aware_retriever() ek smart retriever banata hai jo:Chat history + current question dono ko dekhta hai Sirf relevant documents ko fetch karta hai.
    llm,
    retriever,
    contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant. Use ONLY the information provided in the retrieved context to answer the user's question. "
     "If the answer is not explicitly stated in the context, say 'I don’t know based on the provided documents.' "
     "Do NOT use any outside knowledge or make assumptions.\n\n"
     "Context:\n{context}"),
    
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



##-------chat History with Session ID-------
if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

def get_history(session_id: str):
    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = ChatMessageHistory()
    return st.session_state.chathistory[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


###-----Chat interface-----------
session_id = st.text_input("🆔 Enter Session ID", value="default_session")
user_question = st.chat_input("Ask your question here...")


if user_question:
    history = get_history(session_id)
    result = conversational_rag.invoke(
        {"input": user_question},
        config={"configurable": {"session_id": session_id}},
    )
    answer = result["answer"]


    ##Display chat

    st.chat_message("user").write(user_question)
    st.chat_message("assistant").write(answer)

    # Show chat history
    with st.expander("📖 View Full Chat History"):
        for msg in history.messages:
            role = getattr(msg, "role", msg.type)
            st.markdown(f"**{role.title()}:** {msg.content}") 