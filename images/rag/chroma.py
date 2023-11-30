__import__('pysqlite3')

import json
import sys
import os 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding

import chromadb


# TODO - add sys args 

def load_docs(folder): 
    # get files 
    file_paths = []
    for folder_name, subfolders, files in os.walk(folder):
        for file in files:
            # Create the full path by joining folder path, folder name, and file name
            file_path = os.path.join(folder_name, file)
            file_paths.append(file_path)
    print(f"** Found {len(file_paths)} files ")
    return file_paths   

if len(sys.argv) != 7 : 
    print(f"missing or too much input params expoects 7 got {len(sys.argv)} ")
    print(str(sys.argv))
    exit() 
    
chroma_host = sys.argv[1]
chroma_port = sys.argv[2]
chroma_headers =json.loads( sys.argv[3]) # transform string to dict 
model_name = sys.argv[4]
folder = sys.argv[5]
collection_name = sys.argv[6]

# python chroma.py 'chroma-lightspeed.apps.cn-ai-lab.6aw6.p1.openshiftapps.com' '80' '{"Authorization": "GSe8ipXPZNp4gcVfHixWahi1najVNT6T"}' 'local' ,'./dummy' , 'test'

chroma_client = chromadb.HttpClient(host=chroma_host, port=str(chroma_port), headers=chroma_headers)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name=collection_name,
    get_or_create = True            
    )
print("** Configured Chroma client")


vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
print("** Configured storage_context")

service_context = ServiceContext.from_defaults(chunk_size=512, chunk_overlap=10,embed_model=model_name, llm='local')
print("** Configured service_context")



documents = SimpleDirectoryReader(input_files=load_docs(folder)).load_data()
print("** Loading docs ")

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context, show_progress= True
)

print("*** Completed chromadb embeddings ")