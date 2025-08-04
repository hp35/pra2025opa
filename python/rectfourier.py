#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 09:31:36 2025
Copyright (C) 2025 under GPL 3.0, Fredrik Jonsson
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi

"""
As a global standard, use TeX-style labeling for everything graphics-related.
"""
plt.rcParams.update({
    "text.usetex" : True,
    "font.family" : "Computer Modern",
    "font.size"   : 12
})

Lambda = 2.0
fig, ax = plt.subplots(figsize=(7.0,5.0))
z = np.linspace(0.0, 2.0*Lambda, num=10000)
s = np.zeros_like(z)
for m in range(5):
    if m == 0:
        s += 0.5*np.ones_like(z)
    else:
        s += ((2/pi)/(2*m-1))*np.sin((2*pi*(2*m-1)/Lambda)*z)
    ax.plot(z, s, label='$m$=%d'%m)

ax.autoscale(enable=True, axis='x', tight=True)
ax.legend(loc='upper right')
ax.tick_params(axis="both",direction="in")
ax.grid(visible=True, which='major', axis='both')
ax.set_xlabel("$z$")
ax.set_ylabel("$S(z)$")

kwargs={'bbox_inches':'tight', 'pad_inches':0.0}
fig.savefig("graphs/boxcar.eps", format='eps', **kwargs)
fig.savefig("graphs/boxcar.svg", format='svg', **kwargs)
fig.savefig("graphs/boxcar.png", format='png', **kwargs)
