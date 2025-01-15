import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('loma_code/diff_raytrace_rev.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/diff_raytrace_rev')

    w = 400
    h = 225
    radius = 0.5
    img = np.zeros([h, w, 3], dtype = np.single)
    lib.raytrace(w, h, img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), radius)
    reference = img.copy()

    invN = 1 / (w * h)

    iter = 1000
    losses = []

    radius = 0.3

    lr = 1e-1

    optimization_vis = []
    for i in range(iter):
        lib.raytrace(w, h, img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), radius)
        loss = np.mean((reference - img) ** 2)
        # MD
        dl_dimg = invN * 2 * (img - reference)

        d_radius = lib.diff_raytrace(w, h, dl_dimg.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), radius)

        radius = radius - lr * d_radius

        losses.append(loss)
        optimization_vis.append(radius)

    plt.plot(losses)
    plt.show()
    plt.clf()
    plt.plot(optimization_vis)
    plt.show()