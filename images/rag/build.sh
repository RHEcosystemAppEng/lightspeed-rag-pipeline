#!/bin/bash 

VERSION=44 # 13
IMAGE_REPO=quay.io/ecosystem-appeng


# docker build --pull -f "images/rag/dockerfile" -t rag-embedding:${VERSION} "images/rag"
# docker tag  rag-embedding:${VERSION} ${IMAGE_REPO}/rag-embedding:${VERSION}
# docker push ${IMAGE_REPO}/rag-embedding:${VERSION}

# NOTE: After building image. make sure to update image name `tasks/embedding.yaml` line 66  

docker build --pull -f "images/rag/dockerfile.evaluation" -t rag-evaluation:${VERSION} "images/rag"
docker tag  rag-evaluation:${VERSION} ${IMAGE_REPO}/rag-evaluation:${VERSION}
docker push ${IMAGE_REPO}/rag-evaluation:${VERSION}