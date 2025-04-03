# Equilibrating broad-dataset

This directory contains files relating to equilibrating `broad-dataset.json`. This was done for a couple different force fields and on SLURM/k8s.

Actual Evaluator outputs are not included due to size.

## Running this on Kubernetes

Below are some **incomplete notes** on running this on the NRP.


### Transferring data

Create a bucket.

```
BUCKET_NAME=evaluator-iris-bucket

aws s3api create-bucket --bucket $BUCKET_NAME --profile prp-jm --endpoint-url https://s3-central.nrp-nautilus.io
```

Transfer data from the PVC to the bucket.

```
kubectl apply -f rclone-to-bucket.yaml
```

Copy the data locally.

```
download.sh
```

### Notes on expense

On 30-36 GPUs and 40 CPUs, 857 data points took 2-3 days.