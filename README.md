# Numida
## Important Note:

Installation should be carried out with the versions provided to avoid dependency issues.

## install docker

pip install docker
## Build docker container

docker build -t loan-app .

## Run container

docker run -p 8080:8080 loan-app
## Results
Result is saved in a folder called model_output. The column called "pay_late_predict" contains results of loan behavior in the test data.

