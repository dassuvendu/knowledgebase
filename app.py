import warnings
import doc_upload_vDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import LLMChain
from langchain_community.embeddings import OpenAIEmbeddings
warnings.filterwarnings("ignore", category=DeprecationWarning)


pinecone_api_key = os.getenv("PINECONE_API_KEY")

def storedata(path,indexname):
    pdf_path = path
    documents = doc_upload_vDB.load_pdfs(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    r,add = doc_upload_vDB.vectordb(indexname,texts)
    return r,add


def rag_chat(query,indexname):
    # Initialize Pinecone
    pc = Pinecone()
    index_name = indexname

    # Get the Pinecone index
    index = pc.Index(index_name)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Define LLMs
    rag_llm = ChatOpenAI(temperature=0)
    fallback_llm = ChatOpenAI(temperature=0.7)  # Slightly higher temperature for more general responses

    # RAG template
    rag_template = """You are an AI assistant tasked with answering questions based on provided context information. Your goal is to give accurate, relevant, and concise responses using the given context.
                Here is the context information:
                <context>
                {context}
                </context>

                The user has asked the following question:
                <query>
                {question}
                </query>

                Carefully analyze the context and the query. Look for information in the context that is directly relevant to answering the user's question. If the context doesn't contain information needed to fully answer the query, you may use your general knowledge to supplement, but prioritize information from the given context.

                Formulate a clear and concise response that directly addresses the user's question. Ensure that your answer is:
                1. Accurate and based on the provided context
                2. Relevant to the specific query asked
                3. Concise and to the point
                4. Easy to understand

                If the question cannot be answered based on the context, respond with "FALLBACK_REQUIRED".

                Provide your response in the following format:
                <answer>
                Your response here
                </answer>

                If you used specific information from the context, indicate this by adding a reference note at the end of your answer, like this:
                <reference>
                Information sourced from the provided context.
                </reference>

                Remember, your goal is to provide the most helpful and accurate response possible based on the given context and query."""
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    # Fallback template
    fallback_template = """You are a helpful assistant. The user has asked a question that may be outside the scope of your specific knowledge base. 
    Provide a general response based on your broad knowledge. Be clear that this is a general answer and may not be specific to their exact query.
    Question: {question}
    """
    fallback_prompt = ChatPromptTemplate.from_template(fallback_template)

    # Function to retrieve context from Pinecone
    def retrieve_context(query):
        query_embedding = embeddings.embed_query(query)
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        contexts = [item['metadata']['text'] for item in results['matches'] if 'text' in item['metadata']]
        return "\n".join(contexts)

    # RAG chain
    rag_chain = (
        RunnableParallel({"context": retrieve_context, "question": RunnablePassthrough()})
        | rag_prompt
        | rag_llm
        | StrOutputParser()
    )

    # Fallback chain
    fallback_chain = LLMChain(llm=fallback_llm, prompt=fallback_prompt)

    rag_response = rag_chain.invoke(query)
    if "FALLBACK_REQUIRED" in rag_response:
        fallback_response = fallback_chain.invoke({"question": query})
        
        if isinstance(fallback_response, dict) and 'text' in fallback_response:
            fallback_text = fallback_response['text']
        else:
            fallback_text = str(fallback_response)

        return fallback_text
    return rag_response



# for Rag chat

# q = "expalin Technical Specifications in CB-100 (Basic) ?" # docuoments related question
# # q = "India Prime Minister" # out of documents 
# indexname = "abc"
# e = rag_chat(q,indexname)
# print(e)





# for data store in VectorDB (pinecone)

# path = "pdf"
# indexname = "abc"
# hite,hite2 = storedata(path,indexname)
# print(hite)
# print(hite2)




