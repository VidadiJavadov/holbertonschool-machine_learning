import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backpropagation over a pooling layer.
    """
    m, h_prev, w_prev, c = A_prev.shape
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        da_prev = dA_prev[i]

        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == "max":
                        a_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, ch]
                        mask = (a_slice == np.max(a_slice))
                        da_prev[vert_start:vert_end, horiz_start:horiz_end, ch] += mask * dA[i, h, w, ch]

                    elif mode == "avg":
                        da = dA[i, h, w, ch] / (kh * kw)
                        da_prev[vert_start:vert_end, horiz_start:horiz_end, ch] += np.ones((kh, kw)) * da

        dA_prev[i] = da_prev

    return dA_prev
