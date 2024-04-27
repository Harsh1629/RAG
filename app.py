import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Markdown as md
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

##Creating the chat model and output parser:


##External Source
loader = PyPDFLoader("2404.07143v1.pdf")
pages = loader.load_and_split()

##Creating chunks out of it 
text_splitter = NLTKTextSplitter(chunk_size=200, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

##converting it into embeddings 
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyCO6nFzK0cdfbMebDXQ8e2jDfAF7ZuDcx8", 
                                               model="models/embedding-001")

##storing it in database
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
db.persist()
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
retriever = db_connection.as_retriever(search_kwargs={"k": 5})


##creating the chat template:
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

chat_model = ChatGoogleGenerativeAI(google_api_key="AIzaSyCO6nFzK0cdfbMebDXQ8e2jDfAF7ZuDcx8", 
                                   model="gemini-1.5-pro-latest")

output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)


def main():
    st.title("Leave No Context Behind")
    query = st.text_input("Enter your query:")
    
    if st.button("Submit"):
        st.write(f"You entered: {query}")
    response = rag_chain.invoke(query)
    st.write(md(response))

    
if __name__ == "__main__":
    main()

