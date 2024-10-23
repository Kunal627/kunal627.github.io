# ecg
ECG analysis

This is initial commit

https://github.com/cph-cachet/LocalLeadAttention/tree/main
https://moody-challenge.physionet.org/2020/results/

Datasets
========
ecg Chapman Uni download - https://figshare.com/collections/ChapmanECG/4560497/2


How to run?
==========
1. To start the mlflow server on docker (hosted on localhost:9889)
docker compose build --no-cache
docker compose up -d

run this command on local to get the ip for ftp container
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' vsftpd

2. Update the mlflowrunparams.yaml with run parameters values. These are captured in mlflow db
3. Run the train model script 
4. docker compose down to stop docker containers
