# Running this on Kubernetes


## Transferring data

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
copy.sh
```

## Notes on expense

On 30-36 GPUs and 40 CPUs, 857 data points took 2-3 days.