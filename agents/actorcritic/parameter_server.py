#!/usr/bin/env python
# -*- coding: utf8 -*-

import argparse
import os
import time
import sys
import go_vncdriver  # pylint: disable=W0611
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")))
from misc.utils import cluster_spec  # pylint: disable=C0413

parser = argparse.ArgumentParser()
parser.add_argument("n_tasks", type=int, help="Total number of tasks in this experiment.")
parser.add_argument("--task_id", type=int, default=0, help="ID of this task.")

def main(_):
    args = parser.parse_args()
    spec = cluster_spec(args.n_tasks, 1)
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
