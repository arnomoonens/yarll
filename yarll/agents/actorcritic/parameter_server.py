#!/usr/bin/env python
# -*- coding: utf8 -*-

import argparse
import os
import time
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

from yarll.misc.utils import cluster_spec

parser = argparse.ArgumentParser()
parser.add_argument("n_tasks", type=int, help="Total number of tasks in this experiment.")
parser.add_argument("--n_masters", type=int, default=0, help="Number of masters")
parser.add_argument("--task_id", type=int, default=0, help="ID of this task.")

def main(_):
    args = parser.parse_args()
    spec = cluster_spec(args.n_tasks, 1, args.n_masters)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()
    server = tf.train.Server(  # pylint: disable=W0612
        cluster,
        job_name="ps",
        task_index=args.task_id,
        config=tf.ConfigProto(device_filters=["/job:ps"]))
    while True:
        time.sleep(1000)


if __name__ == '__main__':
    tf.app.run()
