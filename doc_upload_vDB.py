import warnings
import os
import logging
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings


#pinecone 
from pinecone import Pinecone, ServerlessSpec
import time
import uuid
import hashlib
warnings.filterwarnings("ignore", category=DeprecationWarning)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdfs(path):
    documents = []
    
    def process_pdf(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Successfully loaded {pdf_path}")
        except Exception as e:
            logger.error(f"Error loading {pdf_path}: {str(e)}")
    
    if os.path.isfile(path) and path.lower().endswith('.pdf'):
        # Single PDF file
        process_pdf(path)
    elif os.path.isdir(path):
        # Directory of PDF files
        for file in os.listdir(path):
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(path, file)
                process_pdf(pdf_path)
    else:
        logger.error(f"Invalid path: {path}")
    
    return documents


def vectordb(index_name, docs):
    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name in existing_indexes:
        a = f"Index '{index_name}' already exists in the database."
        index = pc.Index(index_name)
    else:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Creating index '{index_name}'...")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        a = f"Index '{index_name}' created successfully."
        index = pc.Index(index_name)

    # Generate unique IDs for each document
    doc_ids = [str(uuid.uuid4()) for _ in range(len(docs))]

    # Create embeddings for the documents
    embedded_docs = embeddings.embed_documents([doc.page_content for doc in docs])

    # Prepare the data for upsert
    to_upsert = []
    for i, (doc, embedding) in enumerate(zip(docs, embedded_docs)):
        # Create a unique hash of the document content
        doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        
        # Check if this hash already exists in the index
        existing_ids = index.query(vector=embedding, top_k=1, include_metadata=True)
        
        if existing_ids['matches'] and existing_ids['matches'][0]['metadata'].get('doc_hash') == doc_hash:
            print(f"Document {i} is a duplicate. Skipping.")
        else:
            to_upsert.append({
                'id': doc_ids[i],
                'values': embedding,
                'metadata': {
                    'doc_hash': doc_hash,
                    'content': doc.page_content,
                    **doc.metadata
                }
            })

    # Upsert the non-duplicate documents
    if to_upsert:
        index.upsert(vectors=to_upsert)
        add = f"{len(to_upsert)} unique document(s) stored in vector database"
    else:
        add = "No new unique documents to store in vector database"

    return a, add




