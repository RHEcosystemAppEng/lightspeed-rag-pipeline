# LightSpeed RAG tetkon pipeline  
This repo contains Tekton pipeline for creation of  LLM embedding using Llama_index 
 
-  Loading data
-  Indexing 
-  Storing
-  Evaluating

<!-- using the following tools 
- [ChromaDB](https://docs.trychroma.com/getting-started) - Simple vectoreDB
- [LLamaindex](https://docs.llamaindex.ai/en/stable/) - RAG framework
- [Milvus](https://milvus.io) - Scalable vectorDB -->


## Prerequisites  
 - Tekton 
 - OpenShift 
 - Nvidia Operator installed 
 - Kostomize CLI
  
## Usage

- Build the embedding container under `images/rag` using the `build.sh` file.
- Add secrets resources to the secret folder and to link the secretes to the pipeline service account 
- Copy and edit the pipeline configuration file `pipeline/data-pipeline.yaml` 
- add you file adn resources to th `kustomization.yaml` file 
- Deploy the pipeline pipeline using the following command: 
```
oc apply -k . 
``` 
- Run the pipeline 
- Results:
    - Container image in the specified location with the embedding json files  
    - GitHub Release zip file which contains the embedding files and metadata files


## Folders 
 - images - base images containing the tekton tasks 


<!-- ## To Do 
- [V] define model in config 
- [V] multiple vector db support: chromadb,llmindex
- [] milvus vector db support 
- [] change defult mode 
- [] add meta data file  -->

## References  
https://docs.llamaindex.ai/en/stable/getting_started/concepts.html
https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms.htm
https://blog.llamaindex.ai/build-and-scale-a-powerful-query-engine-with-llamaindex-and-ray-bfb456404bc4
https://tekton.dev/docs/pipelines/pipelines/ 