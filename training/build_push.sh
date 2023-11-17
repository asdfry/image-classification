#!/bin/bash

docker build -t asdfry/dali-example:imagenet .
yes | docker image prune
docker images
docker push asdfry/dali-example:imagenet
