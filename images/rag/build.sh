#!/bin/bash 

VERSION=1.3

docker build --pull -f "images/rag/dockerfile" -t rag-embedding "images/rag"
docker tag  rag-embedding quay.io/ilan_pinto/rag-embedding
docker push quay.io/ilan_pinto/rag-embedding
