# Numida
## Important Note:

Installation should be carried out with the versions provided to avoid dependency issues. Run the following commands which will install everything.
## create a virtual environment
python -m venv myvenv

## install docker

pip install docker
## Build docker container

docker build -t my-python-app .

## Run container

docker run -v $(pwd)/model_output:/app/model_output my-python-app --input 'train'

## data exploration
The jupyter file dev.ipynb contains step by step exploration of the data .
## Results
Result will be saved in a folder called model_output. The column called "pay_late_predict" contains results of loan behavior in the test data.
The container will run and displa
