import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore as Pinecone
# from langchain.vectorstores import Pinecone
from pinecone import ServerlessSpec, Pinecone as pc
from dotenv import load_dotenv
from crewai_tools import tool

# def getEmbeddings():
#     load_dotenv()
#     embeddings = HuggingFaceEmbeddings()
#     # pinecone.init(
#     #         api_key= os.getenv('PINECONE_API_KEY'),
#     #         environment='gcp-starter'
#     #     )   
#     # Define Index Name
#     # pinecone = Pinecone()
#     # pinecone
#     index_name = "langchain-demo"
#     pinecone = pc(api_key=os.getenv('PINECONE_API_KEY'))
#     # Checking Index
#     if index_name not in pinecone.list_indexes().names():
#         loader = TextLoader('hello_world/self_intro.txt')
#         documents = loader.load()
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
#         docs = text_splitter.split_documents(documents)
#         # Initialize Pinecone client
        
#         # Create new Index
        # pinecone.create_index(name=index_name, metric="cosine", dimension=768, spec=ServerlessSpec(
        #                         cloud="aws",
        #                         region="us-east-1"
        #                     ))
#         docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
#     else:
        
#     # Link to the existing index
#         docsearch = Pinecone(index_name, embeddings)
#     return docsearch

#Tool: get embeddings within vector database
# class GetContext:
# @tool("Get Context Tool")
# def context(question:str) -> str:
#     """Get string input about the question and return the context of the question."""
#     load_dotenv()
#     # pinecone.init(
#     #         api_key= os.getenv('PINECONE_API_KEY'),
#     #         environment='gcp-starter'
#     #     )   
#     # Define Index Name
#     # pinecone = Pinecone()
#     # pinecone
#     embeddings = HuggingFaceEmbeddings()
#     index_name = "langchain-demo"
#     pinecone = pc(api_key=os.getenv('PINECONE_API_KEY'))
#     idx = pinecone.Index(index_name)
#     docsearch = Pinecone(idx, embeddings)
#     return docsearch.similarity_search(question)

# loader = TextLoader('hello_world/self_intro.txt')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=2, separator='\n')
# docs = text_splitter.split_documents(documents)
# print(docs)
# embeddings = HuggingFaceEmbeddings()
# index_name = "langchain-demo"
# pinecone = pc(api_key=os.getenv('PINECONE_API_KEY'))
# # pinecone.create_index(name=index_name, metric="cosine", dimension=768, spec=ServerlessSpec(
# #                                 cloud="aws",
# #                                 region="us-east-1"
# #                             ))
# idx = pinecone.Index(index_name)
# pc = Pinecone(idx, embeddings)

# print(pc.similarity_search("who are you"))
# pc.add_documents(docs)

def context():
        load_dotenv()
        # pinecone.init(
        #         api_key= os.getenv('PINECONE_API_KEY'),
        #         environment='gcp-starter'
        #     )   
        # Define Index Name
        # pinecone = Pinecone()
        # pinecone
        embeddings = HuggingFaceEmbeddings()
        index_name = "langchain-demo"
        pinecone = pc(api_key=os.getenv('PINECONE_API_KEY'))
        idx = pinecone.Index(index_name)
        docsearch = Pinecone(idx, embeddings)
        return docsearch.as_retriever()

# print(context("who are you"))

def similarity_search(query: str):
        load_dotenv()
        # pinecone.init(
        #         api_key= os.getenv('PINECONE_API_KEY'),
        #         environment='gcp-starter'
        #     )   
        # Define Index Name
        # pinecone = Pinecone()
        # pinecone
        embeddings = HuggingFaceEmbeddings()
        index_name = "langchain-demo"
        pinecone = pc(api_key=os.getenv('PINECONE_API_KEY'))
        idx = pinecone.Index(index_name)
        docsearch = Pinecone(idx, embeddings)
        return docsearch.similarity_search(query=query)

print(similarity_search("what is your favourite colour?"))