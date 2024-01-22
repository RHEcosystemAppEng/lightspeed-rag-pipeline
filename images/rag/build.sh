#!/bin/bash 

VERSION=latest # 13
IMAGE_REPO=quay.io/redhat_emp1


docker build --pull -f "images/rag/dockerfile" -t rag-pipeline-tasks:${VERSION} "images/rag"
docker tag  rag-pipeline-tasks:${VERSION} ${IMAGE_REPO}/rag-pipeline-tasks:${VERSION}
docker push ${IMAGE_REPO}/rag-pipeline-tasks:${VERSION}