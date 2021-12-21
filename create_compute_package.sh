#!/bin/bash

mkdir package

FILE_NAME=compute_package.tar.gz

tar -czvf package/$FILE_NAME client

echo done