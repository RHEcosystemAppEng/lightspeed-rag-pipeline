#!/bin/bash 

VERSION=1.3

docker build --pull -f "images/embedding/dockerfile" -t rag-embedding "images/embedding"
docker tag  rag-embedding quay.io/ilan_pinto/rag-embedding
docker push quay.io/ilan_pinto/rag-embedding

cd .. && oc apply -k.