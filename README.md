# Numida
## Important Note:

Installation should be carried out with the versions provided to avoid dependency issues.

## install docker

pip install docker
## Build docker container

docker build -t loan-app .

## Run container

docker run -v $(pwd)/model_output:/app/model_output my-python-app --input 'train'
## Results
Result is saved in a folder called model_output. The column called "pay_late_predict" contains results of loan behavior in the test data.

