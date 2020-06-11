from time import time

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.framework.ops import disable_eager_execution

# tf.compat.v1.disable_eager_execution()
disable_eager_execution()

model_name = "resnet18"
model = tf.keras.models.load_model(f"models/{model_name}/{model_name}-full.h5")
sess = K.get_session()

shape = model.input.shape[1:]
img = np.ones((1, *shape))
img_const = tf.constant(img, dtype=tf.float32)

out_shape = model.output.shape[1:]
output_store = tf.Variable(tf.zeros((1, *out_shape)), name="output_store")
infer_op = output_store.assign(model(img_const))

_ = sess.run([infer_op])

num_samples = 100
t1 = time()
for i in range(num_samples):
    _ = sess.run([infer_op])
t2 = time()

t = (t2 - t1) / num_samples
print(f"{t * 1000:.3f} ms")
