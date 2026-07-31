def compute_2d_objective(rho, u, v, p, dx=1.0, dy=1.0, weights=None):
    import numpy as np
    from scipy.ndimage import median_filter, binary_dilation, gaussian_filter
    from skimage.morphology import skeletonize  # pip install scikit-image
    # --- shock detector ---
    # central differences
    rho_x = (np.pad(rho, ((1,1),(0,0)),'edge')[2:,:] - np.pad(rho, ((1,1),(0,0)),'edge')[:-2,:])/(2*dx)
    rho_y = (np.pad(rho, ((0,0),(1,1)),'edge')[:,2:] - np.pad(rho, ((0,0),(1,1)),'edge')[:,:-2])/(2*dy)
    G = np.sqrt(rho_x**2 + rho_y**2)
    Gmax = np.max(G) + 1e-16
    alpha = 0.10  # threshold fraction of max gradient; tweakable
    S = (G > alpha*Gmax).astype(np.uint8)
    # thin to curve
    try:
        C = skeletonize(S>0)
    except Exception:
        # fallback: use S as curve
        C = S>0

    # dilate to band
    band = binary_dilation(S, structure=np.ones((5,5)))  # 5x5 band; tune width

    # --- shock curve irregularity: std of distance to best-fit line (fast surrogate) ---
    ys, xs = np.where(C)
    if len(xs) < 10:
        # no clear shock detected: set neutral small values
        I_curve = 0.0
    else:
        # fit best straight line x = a*y + b (or orthogonal regression)
        A = np.vstack([ys, np.ones_like(ys)]).T
        a,b = np.linalg.lstsq(A, xs, rcond=None)[0]
        xs_fit = a*ys + b
        I_curve = np.std(xs - xs_fit)  # pixel units

    # --- transverse energy ---
    # local normals from gradient (avoid division by zero)
    nx = rho_x; ny = rho_y
    mag = np.sqrt(nx*nx + ny*ny) + 1e-16
    nxn = nx / mag; nyn = ny / mag
    # tangent
    tx = -nyn; ty = nxn
    vperp = u*tx + v*ty
    E_perp = np.sum((vperp[band])**2)

    # --- Gibbs / overshoot ---
    smooth = median_filter(rho, size=7)  # robust smoothing
    delta = rho - smooth
    if np.any(band):
        O = np.max(np.abs(delta[band])) / (np.max(rho)-np.min(rho)+1e-12)
        E_hf = np.sum((delta[band])**2)
    else:
        O = 0.0
        E_hf = 0.0

    # --- total variation (global or band-limited) ---
    dx_r = np.diff(rho, axis=0); dy_r = np.diff(rho, axis=1)
    TV = (np.sum(np.abs(dx_r)) + np.sum(np.abs(dy_r))) / (rho.size)

    # normalization scales (robust)
    I0 = max(1.0, np.sqrt(rho.shape[0]))  # ~grid scale
    E0 = max(1e-8, np.median(np.abs(v[band]))**2 * np.sum(band)) if np.any(band) else 1.0
    TV0 = max(1e-8, np.mean(np.abs(rho)))  # crude

    # normalized components
    cI = I_curve / I0
    cE = E_perp / E0
    cO = O
    cTV = TV / TV0

    if weights is None:
        weights = {"I":1.0, "E":1.0, "O":50.0, "TV":1.0}

    J = weights["I"]*cI + weights["E"]*cE + weights["O"]*cO + weights["TV"]*cTV

    metrics = {"I_curve":I_curve, "E_perp":E_perp, "O":O, "TV":TV,
               "cI":cI, "cE":cE, "cO":cO, "cTV":cTV}
    return float(J), metrics