import numpy as np
import time
import weno_ext      # the compiled module
from WENO_HLLC_2d import HLLC_x  # your python version for comparison; file referenced earlier. :contentReference[oaicite:1]{index=1}

# create test arrays (shape matching your code: (N+1, N, 4))
N = 200
shape = (N+1, N, 4)
ql = np.random.rand(*shape)
qr = np.random.rand(*shape)
fl = np.random.rand(*shape)
fr = np.random.rand(*shape)
gamma = 1.4

# warmup
out_rs = weno_ext.hllc_x_rs(ql, qr, fl, fr, gamma)

# time Rust version
t0 = time.time()
for _ in range(10):
    out_rs = weno_ext.hllc_x_rs(ql, qr, fl, fr, gamma)
t_rs = (time.time() - t0) / 10

# time Python version
t0 = time.time()
for _ in range(10):
    out_py = HLLC_x(ql, qr, fl, fr, gamma)
t_py = (time.time() - t0) / 10

print(f"Rust average: {t_rs:.6f}s, Python average: {t_py:.6f}s, speedup: {t_py / t_rs:.2f}x")