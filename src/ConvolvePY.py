import numpy as np
from theano import function, config, shared, sandbox
import theano.tensor as T

# instantiate 4D tensor for input
inp_T = T.tensor4(name='inp_T')

def conv4D(img, fltr, strd):

    if strd == 2:
        img_dim = (1, 16, 20, 20)
        fltr_dim = (32, 16, 4, 4)
        out_dim = (1, 32, 9, 9)
    else:
        img_dim = (1, 4, 84, 84)
        fltr_dim = (16, 4, 8, 8)
        out_dim = (1, 16, 20, 20)

    img = np.reshape(np.asarray(img, config.floatX), img_dim)
    fltr = np.reshape(np.asarray(fltr, config.floatX), fltr_dim)

    img_sh = shared(img)
    fltr_sh = shared(fltr)

    res = T.nnet.conv.conv2d(inp_T,
                             fltr_sh, 
                             image_shape=img_dim,
                             filter_shape=fltr_dim,
                             subsample=(strd, strd))

    res_f = function([inp_T], res)
    res_np = res_f(img)

    assert res_np.shape == out_dim
    return res_np.ravel().tolist()

# Grab the read pipe
# Grab the write pipe
fout = open('conv_fifo_out', 'w')
fin = open('conv_fifo_in')
print "Pipes have been grabed"

while(True):
    # Read from name pipe
    inp = fin.readline().split(":")
    img = eval(inp[0])
    fltr = eval(inp[1])
    strd = eval(inp[2])
    # print "Python got: " + str(sum(img) + sum(fltr) + strd)

    # Evaluate convolution
    res = conv4D(img, fltr, strd)

    # Write to named pipe
    fout.write(str(res) + "\n")
    fout.flush()