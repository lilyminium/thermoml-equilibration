#!/usr/bin/env bash

BUCKET_NAME=evaluator-iris-bucket

rclone copy --progress nrp-jm:evaluator-iris-bucket/ . --exclude="*.dcd"