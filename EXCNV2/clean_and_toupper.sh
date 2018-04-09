#!/bin/bash

dir=data

for filename in "$dir"/*.out; do
  submitjob -m 10  sed -i 's/chr//' "$filename"
done
