# LightSpeed RAG tetkon pipeline  
This repo contains tekton pipeline for RAG OCP OCP LightSpeed RAG creations 
following this 5 [concepts](https://docs.trychroma.com/) : 

-  loading data
-  indexing 
-  storing
-  querying 
-  evaluating

using the following tools 
- [ChromaDB](https://docs.trychroma.com/getting-started) - Simple vectoreDB
- [LLamaindex](https://docs.llamaindex.ai/en/stable/) - RAG framework
- [Milvus](https://milvus.io) - Scalable vectorDB


## Usage

## Folders 
 - images - base images containing the tekton tasks 


## To Do 
[V] define model in config 
[V] multiple vector db support: chromadb,llmindex
[] milvus vector db support 
[] change defult mode 
[] add meta data file 
[] add evaluation process 

## references  
https://docs.llamaindex.ai/en/stable/getting_started/concepts.html
https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms.htm
https://blog.llamaindex.ai/build-and-scale-a-powerful-query-engine-with-llamaindex-and-ray-bfb456404bc4 
