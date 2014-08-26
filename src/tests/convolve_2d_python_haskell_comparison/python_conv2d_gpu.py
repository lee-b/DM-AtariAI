from theano import function, config, shared
from theano.tensor.signal import conv
import theano.tensor as T
import numpy as np
import time


img = shared(np.array([1 for x in range(84**2)], dtype=np.float32).reshape((84, 84)))
fltr = shared(np.array(range(8**2), dtype=np.float32).reshape((8, 8)))

f = function([], conv.conv2d(img, fltr, 'full'))

res = f()
print("Output has shape", res.shape)
print("And the sum of the results is", np.sum(res))

num_reps = 1000
acc = 0

start_time = time.time()
for i in range(num_reps):
	acc += np.sum(f())
elapsed_time = time.time() - start_time
print("And the sum of the results is", acc)

time_per_rep = elapsed_time / num_reps
print("The time per rep is",  time_per_rep, "seconds")