# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:06:36 2020
fixes stupid fucking tensorflow-gpu memory allocation
@author: Adam
"""

import tensorflow as tf
# allocate ~4GB of GPU memory:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.67)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# allocate memory as necessary
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
