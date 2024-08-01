# Rag Chat ( Retrieval Augmented Generation)

# Technology used -
1. LangChain( it is used to build the LLM application)
2. Pinecone Vector DB ( it used to store the content of pdf in vector format)
3.  GPT 3.5 TURBO for chat RAG implementation
4. For embedding text-embedding ADA-002 is used


# Features -

1. Upload Document in Pinecone DB
2. If user uploads any duplicate document then it will be not reflect in Pinecone DB.
3. User can chat about the uploaded document.
4. If any content is not present in the uploaded file then "Global Search" will be used to gather information
5. RAG Search and Global Search.
6.  Can create your own index in Pinecone Vector DB.

   
# How to run the application - 

1. Clone the GitHub repo in your machine.
2. Install the requirements.txt file
3. Create a .env file in the project directory. Inside the .env file add your PINECONE API KEY and OPENAPI KEY
4. For chatting purpose comment off the below code in app.py file 

for Rag chat

q = "expalin Technical Specifications in CB-100 (Basic) ?" # docuoments related question

q = "India Prime Minister" # out of documents 

indexname = "your index name"

r= rag_chat(q,indexname)

print(r)

5. For storing data in your vector DB comment off the below in app.py and add the required path of your vector DB

path = "data resources path"

indexname = "your index name"

hite,hite2 = storedata(path,indexname)

print(hite)

print(hite2)
