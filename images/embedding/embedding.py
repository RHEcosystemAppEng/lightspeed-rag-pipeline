__import__('pysqlite3')

import json
import sys
import os 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import argparse

import chromadb

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


# python embedding.py --vector-type chromadb --url 'chroma-lightspeed.apps.cn-ai-lab.6aw6.p1.openshiftapps.com'  --port '80' --auth '{"Authorization": "GSe8ipXPZNp4gcVfHixWahi1najVNT6T"}' --model 'local' --folder './dummy' --collection-name ocp
def main(): 
    # collect args 
    parser = argparse.ArgumentParser(
        description="embedding cli for task execution"
    )
    parser.add_argument("-t", "--vector-type",   default="local", help="Type of vector db[local,chromadb,chromadb-local, milvus]")
    parser.add_argument("-u", "--url", help="VectorDB URL")    
    parser.add_argument("-p", "--port", help="VectorDB port")    
    parser.add_argument("-a", "--auth", help="Authentication headers per vectorDB requirements")    
    parser.add_argument("-n", "--collection-name", help="Collection name in vector DB")
    parser.add_argument("-f", "--folder", help="Plain text folder path")
    parser.add_argument("-m", "--model",   default="local", help="LLM model used for embeddings [local,llama2, or any other supported by llama_index]")
    parser.add_argument("-o", "--output", help="persist folder")


    # execute 
    args = parser.parse_args() 
    
    PERSIST_FOLDER = args.output
    
    # setup storage context 
    if args.vector_type == "local": 
        print("** Local embeddings ") 
        storage_context = StorageContext.from_defaults( )
        
    elif args.vector_type == "chromadb":
        print("** chromadb embeddings ")
        #Validate Inputs 
        if args.url is None or args.port is None: 
            print(" Missing URL or PORT ")
            return 
        
        chroma_client = chromadb.HttpClient(host=args.url, port=args.port, headers=json.loads(args.auth) )
        collection = chroma_client.create_collection(
            name=args.collection_name,
            get_or_create = True            
            )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        
    elif args.vector_type == "chromadb-local":
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(
            name=args.collection_name,
            get_or_create = True            
            )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store )
    
        
    elif args.vector_type == "milvus":
        return 
    
    print("** Configured storage context")
    
    service_context = ServiceContext.from_defaults(chunk_size=512, chunk_overlap=10,embed_model=args.model, llm='local')
    print("** Configured service_context")
    
    documents = SimpleDirectoryReader(input_files=load_docs(args.folder)).load_data()
    print("** Loading docs ")

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context, show_progress= True
    )
    
    index.storage_context.persist(persist_dir=PERSIST_FOLDER)

    print("*** Completed  embeddings ")

main()


