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
    with open('loma_code/diff_raytrace_costlandscape.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/diff_raytrace_cl')

    w = 400
    h = 225
    radius = 0.5
    img = np.zeros([h, w, 3], dtype = np.single)
    lib.raytrace(w, h, img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), radius)
    reference = img.copy()

    iter = 10
    loss = []
    loop = np.linspace(radius - 0.2, radius + 0.2, iter + 1)
    for ri in loop:
        lib.raytrace(w, h, img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), ri)
        plt.imshow(img)
        plt.show()
        loss.append(np.mean((reference - img) ** 2))

    plt.plot(loop, loss)
    plt.show()