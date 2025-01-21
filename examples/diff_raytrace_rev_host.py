import os
import sys

from matplotlib.animation import FuncAnimation

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
    dw = ctypes.c_int(0)
    dh = ctypes.c_int(0)
    radius = 0.5
    img = np.zeros([h, w, 3], dtype = np.single)
    img_denoised = np.zeros([h, w, 3], dtype = np.single)
    lib.raytrace(w, h, img_denoised.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), radius, img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))

    reference = img_denoised.copy()

    invN = 1 / (w * h)

    iters = 1000
    losses = []

    radius = 0.3

    lr = 1e-2

    image_array = []

    optimization_vis = []
    for i in range(iters):
        lib.raytrace(w, h, img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), radius, img_denoised.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))
        image_array.append(img_denoised.copy())
        loss = np.mean((reference - img_denoised) ** 2)
        # MD
        dl_dimg = invN * 2 * (img_denoised - reference)

        dl_dpimg = np.zeros_like(dl_dimg)

        d_radius = lib.diff_raytrace(w, h, img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), dl_dimg.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), radius, dl_dpimg.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))

        radius = radius - lr * d_radius

        losses.append(loss)
        optimization_vis.append(radius)

    fig, ax = plt.subplots(1, 2)
    img_display= ax[0].imshow(np.zeros([h, w, 3]))  # 显示第一张图片
    ax[0].axis("off")
    loss_plt, = ax[1].plot([], [])
    ax[1].set_xlim(0, iters+1)
    ax[1].set_ylim(0, 1.2 * np.max(np.array(losses)))
    def update(frame):
        img_display.set_data(image_array[frame])  # 更新为第 frame 张图片
        loss_plt.set_data(range(frame+1), losses[0:frame+1])
        return img_display, loss_plt

    ani = FuncAnimation(
        fig, update, frames=len(image_array), interval=100, blit=True
    )

    ani.save('inverse_rendering.mp4', fps=40)

    plt.plot(losses)
    plt.show()
    plt.clf()
    plt.plot(optimization_vis)
    plt.show()