from scipy import signal
import numpy as np
import time


img = np.array([1 for x in range(84**2)]).reshape((84, 84))
fltr = np.array(range(8**2)).reshape((8, 8))

res = signal.convolve2d(img, fltr, 'same')
print("Output has shape", res.shape)
print("And the sum of the results is", np.sum(res))

num_reps = 1000
acc = 0

start_time = time.time()
for i in range(num_reps):
	res = signal.convolve2d(img, fltr, 'same')
	acc += np.sum(res)
elapsed_time = time.time() - start_time
print("The sum of the results is", acc)


time_per_rep = elapsed_time / num_reps
print("The time per rep is",  time_per_rep, "seconds")