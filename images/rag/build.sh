#!/bin/bash 

VERSION=14 # 13
IMAGE_REPO="quay.io/ilan_pinto"


docker build --pull -f "images/rag/dockerfile" -t rag-embedding:${VERSION} "images/rag"
docker tag  rag-embedding:${VERSION} ${IMAGE_REPO}/rag-embedding:${VERSION}
docker push ${IMAGE_REPO}/rag-embedding:${VERSION}

# NOTE: After building image. make sure to update image name `tasks/embedding.yaml` line 66  
