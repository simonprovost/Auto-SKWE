#!/bin/bash

echo "Cleaning OutputAutoML"
cd outputAutoML

rm -rf */*.log
rm -rf */*.useless

cd ..

echo "Cleaning outputCrossValidationAutoML"
cd outputCrossValidationAutoML

rm -rf */*.pkl
rm -rf */*.useless

echo "Done"
