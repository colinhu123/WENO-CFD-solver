use ndarray::{ArrayView3, Array3};
use crate::utils;
 
// ---------------------------------------------------------------------------
// Roe-average helpers
// ---------------------------------------------------------------------------
 
/// Compute the Roe-averaged primitive state between two conserved-variable
/// cells for the **x-direction** Riemann problem.
///
/// Returns `(rho_roe, u_roe, v_roe, H_roe, c_roe)`.
#[inline]
fn roe_avg_x(ul: &[f64; 4], ur: &[f64; 4], gamma: f64)
    -> (f64, f64, f64, f64, f64)
{
    // Clamp densities: reconstruction can produce slightly negative rho near
    // vacuum regions; a floor here prevents sqrt(NaN) and 1/0 blow-ups.
    let rho_l = ul[0].max(utils::RHO_MIN);
    let rho_r = ur[0].max(utils::RHO_MIN);
    let sqrt_l = rho_l.sqrt();
    let sqrt_r = rho_r.sqrt();
    let denom  = sqrt_l + sqrt_r;
 
    let u_l = ul[1] / rho_l;
    let v_l = ul[2] / rho_l;
    let u_r = ur[1] / rho_r;
    let v_r = ur[2] / rho_r;
 
    // Total enthalpy  H = (E + p) / rho.
    // Clamp pressure to a small positive value before computing H so that
    // H stays bounded even when the energy is slightly negative due to
    // reconstruction overshoot.
    let p_l = ((gamma - 1.0) * (ul[3] - 0.5 * rho_l * (u_l*u_l + v_l*v_l)))
                  .max(utils::P_MIN);
    let p_r = ((gamma - 1.0) * (ur[3] - 0.5 * rho_r * (u_r*u_r + v_r*v_r)))
                  .max(utils::P_MIN);
    let h_l = (ul[3] + p_l) / rho_l;
    let h_r = (ur[3] + p_r) / rho_r;
 
    let u_roe = (sqrt_l * u_l + sqrt_r * u_r) / denom;
    let v_roe = (sqrt_l * v_l + sqrt_r * v_r) / denom;
    let h_roe = (sqrt_l * h_l + sqrt_r * h_r) / denom;
 
    let kin   = 0.5 * (u_roe*u_roe + v_roe*v_roe);
    // Guarantee c² > 0: if H_roe - kin goes negative (can happen with
    // severely under-resolved shocks), fall back to a minimum sound speed.
    let c2    = ((gamma - 1.0) * (h_roe - kin)).max(utils::P_MIN / rho_l.min(rho_r));
    let c_roe = c2.sqrt();
    let rho_roe = sqrt_l * sqrt_r; // geometric mean, always positive
 
    (rho_roe, u_roe, v_roe, h_roe, c_roe)
}
 
/// Same as `roe_avg_x` but for the **y-direction** Riemann problem.
/// The role of u and v are swapped in the eigenvector matrices.
#[inline]
fn roe_avg_y(ul: &[f64; 4], ur: &[f64; 4], gamma: f64)
    -> (f64, f64, f64, f64, f64)
{
    // Identical arithmetic; labelling is purely cosmetic here because the
    // eigenvectors below will use the correct velocity components.
    roe_avg_x(ul, ur, gamma)
}
 
// ---------------------------------------------------------------------------
// Eigenvector matrices – 2D Euler equations
// ---------------------------------------------------------------------------
 
/// Left eigenvector matrix L for the **x-direction** flux Jacobian.
///
/// Rows of L transform conserved differences into characteristic differences:
///   dW = L · dU
///
/// Ordering of eigenvalues: (u-c, u, u, u+c)
#[inline]
fn left_eigenvectors_x(rho: f64, u: f64, v: f64, c: f64, gamma: f64)
    -> [[f64; 4]; 4]
{
    let g1  = gamma - 1.0;
    let kin = 0.5 * (u*u + v*v);
    let rc  = rho * c;
    let inv_rc = 1.0 / rc;
    let inv_c2 = 1.0 / (c * c);
 
    [
        // row 0  ->  eigenvalue (u - c)
        [ g1 * kin / (2.0 * c * c) + u / (2.0 * c),
         -g1 * u * inv_c2 / 2.0   - 0.5 * inv_rc,
         -g1 * v * inv_c2 / 2.0,
          g1 * inv_c2 / 2.0 ],
        // row 1  ->  eigenvalue u  (entropy / density wave)
        [ 1.0 - g1 * kin * inv_c2,
          g1 * u * inv_c2,
          g1 * v * inv_c2,
         -g1 * inv_c2 ],
        // row 2  ->  eigenvalue u  (shear / v-momentum wave)
        [ -v / rho,
          0.0,
          1.0 / rho,
          0.0 ],
        // row 3  ->  eigenvalue (u + c)
        [ g1 * kin / (2.0 * c * c) - u / (2.0 * c),
         -g1 * u * inv_c2 / 2.0   + 0.5 * inv_rc,
         -g1 * v * inv_c2 / 2.0,
          g1 * inv_c2 / 2.0 ],
    ]
}
 
/// Right eigenvector matrix R for the **x-direction** flux Jacobian.
/// Columns of R transform characteristic variables back to conserved space:
///   dU = R · dW
#[inline]
fn right_eigenvectors_x(rho: f64, u: f64, v: f64, h: f64, c: f64)
    -> [[f64; 4]; 4]
{
    let kin = 0.5 * (u*u + v*v);

    [
        // rho row
        [1.0, 1.0, 0.0, 1.0],

        // rho*u row
        [u - c, u, 0.0, u + c],

        // rho*v row
        [v, v, rho, v],

        // E row
        [h - u*c, kin, rho*v, h + u*c],
    ]
}
 
/// Left eigenvector matrix L for the **y-direction** flux Jacobian.
/// Eigenvalue ordering: (v-c, v, v, v+c).
#[inline]
fn left_eigenvectors_y(rho: f64, u: f64, v: f64, c: f64, gamma: f64)
    -> [[f64; 4]; 4]
{
    let g1  = gamma - 1.0;
    let kin = 0.5 * (u*u + v*v);
    let rc  = rho * c;
    let inv_rc = 1.0 / rc;
    let inv_c2 = 1.0 / (c * c);
 
    [
        // row 0  ->  eigenvalue (v - c)
        [ g1 * kin / (2.0 * c * c) + v / (2.0 * c),
         -g1 * u * inv_c2 / 2.0,
         -g1 * v * inv_c2 / 2.0   - 0.5 * inv_rc,
          g1 * inv_c2 / 2.0 ],
        // row 1  ->  eigenvalue v  (entropy wave)
        [ 1.0 - g1 * kin * inv_c2,
          g1 * u * inv_c2,
          g1 * v * inv_c2,
         -g1 * inv_c2 ],
        // row 2  ->  eigenvalue v  (shear / u-momentum wave)
        [ -u / rho,
          1.0 / rho,
          0.0,
          0.0 ],
        // row 3  ->  eigenvalue (v + c)
        [ g1 * kin / (2.0 * c * c) - v / (2.0 * c),
         -g1 * u * inv_c2 / 2.0,
         -g1 * v * inv_c2 / 2.0   + 0.5 * inv_rc,
          g1 * inv_c2 / 2.0 ],
    ]
}
 
/// Right eigenvector matrix R for the **y-direction** flux Jacobian.
#[inline]
fn right_eigenvectors_y(rho: f64, u: f64, v: f64, h: f64, c: f64)
    -> [[f64; 4]; 4]
{
    let kin = 0.5 * (u*u + v*v);

    [
        // rho row
        [1.0, 1.0, 0.0, 1.0],

        // rho*u row
        [u, u, rho, u],

        // rho*v row
        [v - c, v, 0.0, v + c],

        // E row
        [h - v*c, kin, rho*u, h + v*c],
    ]
}
 
// ---------------------------------------------------------------------------
// Matrix–vector helpers
// ---------------------------------------------------------------------------
 
/// Multiply a 4×4 matrix by a length-4 vector.
#[inline]
fn mat4_vec4(m: &[[f64; 4]; 4], v: &[f64; 4]) -> [f64; 4] {
    let mut out = [0.0f64; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r] += m[r][c] * v[c];
        }
    }
    out
}
 
// ---------------------------------------------------------------------------
// WENO weights (scalar, component-wise)
// ---------------------------------------------------------------------------
 
/// WENO-JS smoothness indicators and weights for a 5-point left-biased stencil
/// (reconstructing the value at the **right** face from the left side).
#[inline]
fn weno5_left(u0: f64, u1: f64, u2: f64, u3: f64, u4: f64) -> f64 {
    let beta0 = 13.0/12.0 * (u2 - 2.0*u3 + u4).powi(2)
                + 0.25 * (3.0*u2 - 4.0*u3 + u4).powi(2);
    let beta1 = 13.0/12.0 * (u1 - 2.0*u2 + u3).powi(2)
                + 0.25 * (u1 - u3).powi(2);
    let beta2 = 13.0/12.0 * (u0 - 2.0*u1 + u2).powi(2)
                + 0.25 * (u0 - 4.0*u1 + 3.0*u2).powi(2);
 
    let d0 = 0.3; let d1 = 0.6; let d2 = 0.1;
    let a0 = d0 / (utils::DEFAULT_EPS + beta0).powi(2);
    let a1 = d1 / (utils::DEFAULT_EPS + beta1).powi(2);
    let a2 = d2 / (utils::DEFAULT_EPS + beta2).powi(2);
    let s  = a0 + a1 + a2;
    let (w0, w1, w2) = (a0/s, a1/s, a2/s);
 
    let p0 = -u4 + 5.0*u3 + 2.0*u2;
    let p1 = -u1 + 5.0*u2 + 2.0*u3;
    let p2 =  2.0*u0 - 7.0*u1 + 11.0*u2;
    (w0*p0 + w1*p1 + w2*p2) / 6.0
}
 
/// WENO-JS weights for a 5-point right-biased stencil
/// (reconstructing the value at the **left** face from the right side).
#[inline]
fn weno5_right(u1: f64, u2: f64, u3: f64, u4: f64, u5: f64) -> f64 {
    let beta0 = 13.0/12.0 * (u3 - 2.0*u4 + u5).powi(2)
                + 0.25 * (3.0*u3 - 4.0*u4 + u5).powi(2);
    let beta1 = 13.0/12.0 * (u2 - 2.0*u3 + u4).powi(2)
                + 0.25 * (u2 - u4).powi(2);
    let beta2 = 13.0/12.0 * (u1 - 2.0*u2 + u3).powi(2)
                + 0.25 * (u1 - 4.0*u2 + 3.0*u3).powi(2);
 
    let d0 = 0.1; let d1 = 0.6; let d2 = 0.3;
    let a0 = d0 / (utils::DEFAULT_EPS + beta0).powi(2);
    let a1 = d1 / (utils::DEFAULT_EPS + beta1).powi(2);
    let a2 = d2 / (utils::DEFAULT_EPS + beta2).powi(2);
    let s  = a0 + a1 + a2;
    let (w0, w1, w2) = (a0/s, a1/s, a2/s);
 
    let p0 = 2.0*u5 - 7.0*u4 + 11.0*u3;
    let p1 = -u4 + 5.0*u3 + 2.0*u2;
    let p2 = -u1 + 5.0*u2 + 2.0*u3;
    (w0*p0 + w1*p1 + w2*p2) / 6.0
}
 
// Gaussian-quadrature WENO variants (used for the transverse direction).
 
#[inline]
fn weno5_gq_left_a(q0: f64, q1: f64, q2: f64, q3: f64, q4: f64) -> f64 {
    let beta0 = 13.0/12.0 * (q2 - 2.0*q3 + q4).powi(2)
                + 0.25 * (3.0*q2 - 4.0*q3 + q4).powi(2);
    let beta1 = 13.0/12.0 * (q1 - 2.0*q2 + q3).powi(2)
                + 0.25 * (q1 - q3).powi(2);
    let beta2 = 13.0/12.0 * (q0 - 2.0*q1 + q2).powi(2)
                + 0.25 * (q0 - 4.0*q1 + 3.0*q2).powi(2);
 
    let d0 = 0.1928406937; let d1 = 0.6111111111; let d2 = 0.1960481952;
    let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
    let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
    let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);
    let s = a0+a1+a2;
    let (w0,w1,w2) = (a0/s,a1/s,a2/s);
 
    let p0 = q2 + (3.0*q2 - 4.0*q3 + q4)*utils::SQRT3_12;
    let p1 = q2 - (-q1 + q3)*utils::SQRT3_12;
    let p2 = q2 - (3.0*q2 - 4.0*q1 + q0)*utils::SQRT3_12;
    w0*p0 + w1*p1 + w2*p2
}
 
#[inline]
fn weno5_gq_left_b(q0: f64, q1: f64, q2: f64, q3: f64, q4: f64) -> f64 {
    let beta0 = 13.0/12.0 * (q2 - 2.0*q3 + q4).powi(2)
                + 0.25 * (3.0*q2 - 4.0*q3 + q4).powi(2);
    let beta1 = 13.0/12.0 * (q1 - 2.0*q2 + q3).powi(2)
                + 0.25 * (q1 - q3).powi(2);
    let beta2 = 13.0/12.0 * (q0 - 2.0*q1 + q2).powi(2)
                + 0.25 * (q0 - 4.0*q1 + 3.0*q2).powi(2);
 
    let d0 = 0.1960481952; let d1 = 0.6111111111; let d2 = 0.1928406937;
    let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
    let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
    let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);
    let s = a0+a1+a2;
    let (w0,w1,w2) = (a0/s,a1/s,a2/s);
 
    let p0 = q2 - (3.0*q2 - 4.0*q3 + q4)*utils::SQRT3_12;
    let p1 = q2 - (q1 - q3)*utils::SQRT3_12;
    let p2 = q2 + (3.0*q2 - 4.0*q1 + q0)*utils::SQRT3_12;
    w0*p0 + w1*p1 + w2*p2
}
 
// ---------------------------------------------------------------------------
// Zhang & Shu (2010) positivity-preserving limiter
// ---------------------------------------------------------------------------
 
/// Compute pressure from a conserved state.  Returns 0 if density is
/// non-positive so the limiter can still act on the result.
#[inline]
fn pressure(q: &[f64; 4], gamma: f64) -> f64 {
    if q[0] <= 0.0 { return 0.0; }
    (gamma - 1.0) * (q[3] - 0.5 * (q[1]*q[1] + q[2]*q[2]) / q[0])
}
 
/// Zhang & Shu two-pass positivity-preserving limiter.
///
/// Given the cell average `u_bar` and two reconstructed face values
/// `ql` (left face) and `qr` (right face) for that same cell, scale
/// both face perturbations toward the cell average until:
///   Pass 1 – all face densities ≥ RHO_MIN
///   Pass 2 – all face pressures ≥ P_MIN
///
/// The cell average is never modified, so conservation is preserved exactly.
#[inline]
fn  pp_limit(q: &mut [f64; 4], u_bar: &[f64; 4], gamma: f64) {
    // density
    if q[0] < utils::RHO_MIN {
        let rho_bar = u_bar[0].max(utils::RHO_MIN);
        let theta = ((rho_bar - utils::RHO_MIN) / (rho_bar - q[0]))
            .min(1.0)
            .max(0.0);

        for k in 0..4 {
            q[k] = theta * (q[k] - u_bar[k]) + u_bar[k];
        }
    }

    // pressure
    let p_q = pressure(q, gamma);
    if p_q < utils::P_MIN {
        let p_bar = pressure(u_bar, gamma).max(utils::P_MIN);
        let theta = ((p_bar - utils::P_MIN) / (p_bar - p_q))
            .min(1.0)
            .max(0.0);

        for k in 0..4 {
            q[k] = theta * (q[k] - u_bar[k]) + u_bar[k];
        }
    }
}
 
// ---------------------------------------------------------------------------
// Extract a length-4 conserved state from a 3D array cell
// ---------------------------------------------------------------------------
 
#[inline]
fn get_cell(arr: &Array3<f64>, i: usize, j: usize) -> [f64; 4] {
    [arr[[i,j,0]], arr[[i,j,1]], arr[[i,j,2]], arr[[i,j,3]]]
}
 
// ---------------------------------------------------------------------------
// Characteristic WENO reconstruction – x-direction
// ---------------------------------------------------------------------------
 
pub(crate) fn weno_x_reconstruct_local(u: ArrayView3<'_, f64>, gamma: f64)
    -> (Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>)
{
    let (m, n, c) = u.dim();
    assert!(c >= 4, "u must have at least 4 channels");
 
    // Copy the first 4 conservative variables.
    let mut u_con = Array3::<f64>::zeros((m, n, 4));
    for i in 0..m {
        for j in 0..n {
            for k in 0..4 { u_con[[i,j,k]] = u[[i,j,k]]; }
        }
    }
 
    // -----------------------------------------------------------------------
    // Pass 1 – x-direction WENO with characteristic projection
    //          Stencil width 6 → output has out_m = m-5 rows.
    // -----------------------------------------------------------------------
    let out_m = m - 5;
    let out_n = n;
    let mut q_l = Array3::<f64>::zeros((out_m, out_n, 4));
    let mut q_r = Array3::<f64>::zeros((out_m, out_n, 4));
 
    for i in 0..out_m {
        for j in 0..out_n {
            // Collect the 6-cell stencil of conserved states.
            let cells: [[f64; 4]; 6] = [
                get_cell(&u_con, i,   j),
                get_cell(&u_con, i+1, j),
                get_cell(&u_con, i+2, j),
                get_cell(&u_con, i+3, j),
                get_cell(&u_con, i+4, j),
                get_cell(&u_con, i+5, j),
            ];
 
            // cons_l is the right face of cell i+2  →  Roe avg at cells[2|3]
            // cons_r is the left  face of cell i+3  →  Roe avg at cells[3|4]
            // Each face gets its own eigenvector basis so that a discontinuity
            // sitting on one interface does not corrupt the other face's
            // projection, which was the cause of the single-column stripe.
 
            // ---- Left face (right face of cell i+2) -----------------------
            let (rho_al, u_al, v_al, h_al, c_al) =
                roe_avg_x(&cells[2], &cells[3], gamma);
            let l_mat_l = left_eigenvectors_x(rho_al, u_al, v_al, c_al, gamma);
            let r_mat_l = right_eigenvectors_x(rho_al, u_al, v_al, h_al, c_al);
            let wl: [[f64; 4]; 5] = [
                mat4_vec4(&l_mat_l, &cells[0]),
                mat4_vec4(&l_mat_l, &cells[1]),
                mat4_vec4(&l_mat_l, &cells[2]),
                mat4_vec4(&l_mat_l, &cells[3]),
                mat4_vec4(&l_mat_l, &cells[4]),
            ];
            let mut char_l = [0.0f64; 4];
            for k in 0..4 {
                char_l[k] = weno5_left(wl[0][k], wl[1][k], wl[2][k],
                                       wl[3][k], wl[4][k]);
            }
            let mut cons_l = mat4_vec4(&r_mat_l, &char_l);
 
            // ---- Right face (left face of cell i+3) -----------------------
            let (rho_ar, u_ar, v_ar, h_ar, c_ar) =
                roe_avg_x(&cells[2], &cells[3], gamma);
            let l_mat_r = left_eigenvectors_x(rho_ar, u_ar, v_ar, c_ar, gamma);
            let r_mat_r = right_eigenvectors_x(rho_ar, u_ar, v_ar, h_ar, c_ar);
            let wr: [[f64; 4]; 5] = [
                mat4_vec4(&l_mat_r, &cells[1]),
                mat4_vec4(&l_mat_r, &cells[2]),
                mat4_vec4(&l_mat_r, &cells[3]),
                mat4_vec4(&l_mat_r, &cells[4]),
                mat4_vec4(&l_mat_r, &cells[5]),
            ];
            let mut char_r = [0.0f64; 4];
            for k in 0..4 {
                char_r[k] = weno5_right(wr[0][k], wr[1][k], wr[2][k],
                                        wr[3][k], wr[4][k]);
            }
            let mut cons_r = mat4_vec4(&r_mat_r, &char_r);
 
            // Zhang-Shu limiter with the correct cell average for each face.
            pp_limit(&mut cons_l, &cells[2], gamma);
            pp_limit(&mut cons_r, &cells[3], gamma);
            for k in 0..4 {
                q_l[[i,j,k]] = cons_l[k];
                q_r[[i,j,k]] = cons_r[k];
            }
        }
    }
 
    // -----------------------------------------------------------------------
    // Pass 2 – y-direction Gauss-quadrature WENO on the x-reconstructed values
    //          Stencil width 5 (indices j+1..j+5) → output n1 = out_n-6.
    // -----------------------------------------------------------------------
    let m1 = out_m;
    let n1 = out_n - 6;
 
    let mut q_l1 = Array3::<f64>::zeros((m1, n1, 4));
    let mut q_l2 = Array3::<f64>::zeros((m1, n1, 4));
    let mut q_r1 = Array3::<f64>::zeros((m1, n1, 4));
    let mut q_r2 = Array3::<f64>::zeros((m1, n1, 4));
 
    for i in 0..m1 {
        for j in 0..n1 {
            for k in 0..4 {
                // ---- q_l (left-face states) – component-wise GQ ----------
                let q0 = q_l[[i, j+1, k]];
                let q1 = q_l[[i, j+2, k]];
                let q2 = q_l[[i, j+3, k]];
                let q3 = q_l[[i, j+4, k]];
                let q4 = q_l[[i, j+5, k]];
                q_l1[[i,j,k]] = weno5_gq_left_a(q0, q1, q2, q3, q4);
                q_l2[[i,j,k]] = weno5_gq_left_b(q0, q1, q2, q3, q4);
 
                // ---- q_r (right-face states) – component-wise GQ ---------
                let q0 = q_r[[i, j+1, k]];
                let q1 = q_r[[i, j+2, k]];
                let q2 = q_r[[i, j+3, k]];
                let q3 = q_r[[i, j+4, k]];
                let q4 = q_r[[i, j+5, k]];
                q_r1[[i,j,k]] = weno5_gq_left_a(q0, q1, q2, q3, q4);
                q_r2[[i,j,k]] = weno5_gq_left_b(q0, q1, q2, q3, q4);
            }
        }
    }
 
    let flux_l1 = utils::flux_x_local(q_l1.view(), gamma);
    let flux_l2 = utils::flux_x_local(q_l2.view(), gamma);
    let flux_r1 = utils::flux_x_local(q_r1.view(), gamma);
    let flux_r2 = utils::flux_x_local(q_r2.view(), gamma);
 
    let flux_l = 0.5 * (&flux_l1 + &flux_l2);
    let flux_r = 0.5 * (&flux_r2 + &flux_r1);
    let q_l    = 0.5 * (&q_l1 + &q_l2);
    let q_r    = 0.5 * (&q_r1 + &q_r2);
    (q_l, flux_l, q_r, flux_r)
}
 
// ---------------------------------------------------------------------------
// Characteristic WENO reconstruction – y-direction
// ---------------------------------------------------------------------------
 
pub(crate) fn weno_y_reconstruct_local(u: ArrayView3<'_, f64>, gamma: f64)
    -> (Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>)
{
    let (m, n, c) = u.dim();
    assert!(c >= 4, "u must have at least 4 channels");
 
    let mut u_con = Array3::<f64>::zeros((m, n, 4));
    for i in 0..m {
        for j in 0..n {
            for k in 0..4 { u_con[[i,j,k]] = u[[i,j,k]]; }
        }
    }
 
    // -----------------------------------------------------------------------
    // Pass 1 – y-direction WENO with characteristic projection
    // -----------------------------------------------------------------------
    let out_m = m;
    let out_n = n - 5;
    let mut q_l = Array3::<f64>::zeros((out_m, out_n, 4));
    let mut q_r = Array3::<f64>::zeros((out_m, out_n, 4));
 
    for i in 0..out_m {
        for j in 0..out_n {
            let cells: [[f64; 4]; 6] = [
                get_cell(&u_con, i, j),
                get_cell(&u_con, i, j+1),
                get_cell(&u_con, i, j+2),
                get_cell(&u_con, i, j+3),
                get_cell(&u_con, i, j+4),
                get_cell(&u_con, i, j+5),
            ];
 
            // ---- Left face (right face of cell j+2) -----------------------
            let (rho_al, u_al, v_al, h_al, c_al) =
                roe_avg_y(&cells[2], &cells[3], gamma);
            let l_mat_l = left_eigenvectors_y(rho_al, u_al, v_al, c_al, gamma);
            let r_mat_l = right_eigenvectors_y(rho_al, u_al, v_al, h_al, c_al);
            let wl: [[f64; 4]; 5] = [
                mat4_vec4(&l_mat_l, &cells[0]),
                mat4_vec4(&l_mat_l, &cells[1]),
                mat4_vec4(&l_mat_l, &cells[2]),
                mat4_vec4(&l_mat_l, &cells[3]),
                mat4_vec4(&l_mat_l, &cells[4]),
            ];
            let mut char_l = [0.0f64; 4];
            for k in 0..4 {
                char_l[k] = weno5_left(wl[0][k], wl[1][k], wl[2][k],
                                       wl[3][k], wl[4][k]);
            }
            let mut cons_l = mat4_vec4(&r_mat_l, &char_l);
 
            // ---- Right face (left face of cell j+3) -----------------------
            let (rho_ar, u_ar, v_ar, h_ar, c_ar) =
                roe_avg_y(&cells[2], &cells[3], gamma);
            let l_mat_r = left_eigenvectors_y(rho_ar, u_ar, v_ar, c_ar, gamma);
            let r_mat_r = right_eigenvectors_y(rho_ar, u_ar, v_ar, h_ar, c_ar);
            let wr: [[f64; 4]; 5] = [
                mat4_vec4(&l_mat_r, &cells[1]),
                mat4_vec4(&l_mat_r, &cells[2]),
                mat4_vec4(&l_mat_r, &cells[3]),
                mat4_vec4(&l_mat_r, &cells[4]),
                mat4_vec4(&l_mat_r, &cells[5]),
            ];
            let mut char_r = [0.0f64; 4];
            for k in 0..4 {
                char_r[k] = weno5_right(wr[0][k], wr[1][k], wr[2][k],
                                        wr[3][k], wr[4][k]);
            }
            let mut cons_r = mat4_vec4(&r_mat_r, &char_r);
 
            pp_limit(&mut cons_l, &cells[2], gamma);
            pp_limit(&mut cons_r, &cells[3], gamma);
            for k in 0..4 {
                q_l[[i,j,k]] = cons_l[k];
                q_r[[i,j,k]] = cons_r[k];
            }
        }
    }
 
    // -----------------------------------------------------------------------
    // Pass 2 – x-direction Gauss-quadrature WENO on y-reconstructed values
    // -----------------------------------------------------------------------
    let m1 = out_m - 6;
    let n1 = out_n;
 
    let mut q_l1 = Array3::<f64>::zeros((m1, n1, 4));
    let mut q_l2 = Array3::<f64>::zeros((m1, n1, 4));
    let mut q_r1 = Array3::<f64>::zeros((m1, n1, 4));
    let mut q_r2 = Array3::<f64>::zeros((m1, n1, 4));
 
    for i in 0..m1 {
        for j in 0..n1 {
            for k in 0..4 {
                // ---- q_l – component-wise GQ --------------------------------
                let q0 = q_l[[i+1, j, k]];
                let q1 = q_l[[i+2, j, k]];
                let q2 = q_l[[i+3, j, k]];
                let q3 = q_l[[i+4, j, k]];
                let q4 = q_l[[i+5, j, k]];
                q_l1[[i,j,k]] = weno5_gq_left_a(q0, q1, q2, q3, q4);
                q_l2[[i,j,k]] = weno5_gq_left_b(q0, q1, q2, q3, q4);
 
                // ---- q_r – component-wise GQ --------------------------------
                let q0 = q_r[[i+1, j, k]];
                let q1 = q_r[[i+2, j, k]];
                let q2 = q_r[[i+3, j, k]];
                let q3 = q_r[[i+4, j, k]];
                let q4 = q_r[[i+5, j, k]];
                q_r1[[i,j,k]] = weno5_gq_left_a(q0, q1, q2, q3, q4);
                q_r2[[i,j,k]] = weno5_gq_left_b(q0, q1, q2, q3, q4);
            }
        }
    }
 
    let flux_l1 = utils::flux_y_local(q_l1.view(), gamma);
    let flux_l2 = utils::flux_y_local(q_l2.view(), gamma);
    let flux_r1 = utils::flux_y_local(q_r1.view(), gamma);
    let flux_r2 = utils::flux_y_local(q_r2.view(), gamma);
 
    let flux_l = 0.5 * (&flux_l1 + &flux_l2);
    let flux_r = 0.5 * (&flux_r2 + &flux_r1);
    let q_l    = 0.5 * (&q_l1 + &q_l2);
    let q_r    = 0.5 * (&q_r1 + &q_r2);
    (q_l, flux_l, q_r, flux_r)
}


pub(crate) fn weno_x_reconstruct_local_cons(u: ArrayView3<'_, f64>, gamma: f64) -> 
(Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>)
{
    let (m,n,c) = u.dim();
    assert!(c>= 4, "u must have at least 4 channels");
    let mut u_con = Array3::<f64>::zeros((m,n,4));
    for i in 0..m {
        for j in 0..n {
            for k in 0..4 {
                u_con[[i,j,k]] = u[[i,j,k]];
            }
        }
    }

    let out_m = m-5;
    let out_n = n;
    let mut q_l = Array3::<f64>::zeros((out_m, out_n, 4));
    let mut q_r = Array3::<f64>::zeros((out_m, out_n, 4)); 

    for i in 0..out_m {
        for j in 0..out_n {
            for k in 0..4 {
                let u0 = u_con[[i,j,k]];
                let u1 = u_con[[i+1,j,k]];
                let u2 = u_con[[i+2,j,k]];
                let u3 = u_con[[i+3,j,k]];
                let u4 = u_con[[i+4,j,k]];
                let u5 = u_con[[i+5,j,k]];

                let beta0 = 13.0/12.0 * (u2 - 2.0*u3 + u4).powi(2)
                                + 0.25*(3.0*u2 - 4.0*u3 + u4).powi(2);
                let beta1 = 13.0/12.0*(u1 - 2.0*u2 + u3).powi(2)
                                + 0.25*(u1 - u3).powi(2);
                let beta2 = 13.0/12.0*(u0 - 2.0*u1 + u2).powi(2)
                                + 0.25*(u0 - 4.0*u1 + 3.0*u2).powi(2);
                let d0 = 0.3;
                let d1 = 0.6;
                let d2 = 0.1;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = - u4 + 5.0*u3 + 2.0*u2;
                let p1 = - u1 + 5.0*u2 + 2.0*u3;
                let p2 = 2.0*u0 - 7.0*u1 + 11.0*u2;

                q_l[[i,j,k]] = (w0*p0 + w1*p1 + w2*p2)/6.0;

                let beta0 = 13.0/12.0 * (u3 - 2.0*u4 + u5).powi(2)
                                + 0.25*(3.0*u3 - 4.0*u4 + u5).powi(2);
                let beta1 = 13.0/12.0*(u2 - 2.0*u3 + u4).powi(2)
                                + 0.25*(u2 - u4).powi(2);
                let beta2 = 13.0/12.0*(u1 - 2.0*u2 + u3).powi(2)
                                + 0.25*(u1 - 4.0*u2 + 3.0*u3).powi(2);
                let d0 = 0.1;
                let d1 = 0.6;
                let d2 = 0.3;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = 2.0*u5 - 7.0*u4 + 11.0*u3;
                let p1 = -u4 + 5.0*u3 + 2.0*u2;
                let p2 = -u1 + 5.0*u2 + 2.0*u3;

                q_r[[i,j,k]] = (w0*p0+w1*p1+w2*p2)/6.0;
            }
        }
    }


    let m1 = out_m;
    let n1 = out_n-6;

    let mut q_l1 = Array3::<f64>::zeros((m1,n1,4));
    let mut q_l2 = Array3::<f64>::zeros((m1,n1,4));
    let mut q_r1 = Array3::<f64>::zeros((m1,n1,4));
    let mut q_r2 = Array3::<f64>::zeros((m1,n1,4));

    for i in 0..m1 {
        for j in 0..n1 {
            for k in 0..4 {
                let q0 = q_l[[i,j+1,k]];
                let q1 = q_l[[i,j+2,k]];
                let q2 = q_l[[i,j+3,k]];
                let q3 = q_l[[i,j+4,k]];
                let q4 = q_l[[i,j+5,k]];

                let beta0 = 13.0/12.0 * (q2 - 2.0*q3 + q4).powi(2)
                                + 0.25*(3.0*q2 - 4.0*q3 + q4).powi(2);
                let beta1 = 13.0/12.0*(q1 - 2.0*q2 + q3).powi(2)
                                + 0.25*(q1 - q3).powi(2);
                let beta2 = 13.0/12.0*(q0 - 2.0*q1 + q2).powi(2)
                                + 0.25*(q0 - 4.0*q1 + 3.0*q2).powi(2);

                let d0 = 0.1928406937;
                let d1 = 0.6111111111;
                let d2 = 0.1960481952;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = q2 + (3.0*q2 - 4.0*q3 + q4)*utils::SQRT3_12;
                let p1 = q2 - (-q1 + q3)*utils::SQRT3_12;
                let p2 = q2 - (3.0*q2 - 4.0*q1 + q0)*utils::SQRT3_12;

                q_l1[[i,j,k]] = w0*p0 + w1*p1 + w2*p2;

                let d0 = 0.1960481952;
                let d1 = 0.6111111111;
                let d2 = 0.1928406937;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = q2 - (3.0*q2 - 4.0*q3 + q4)*utils::SQRT3_12;
                let p1 = q2 - (q1 - q3)*utils::SQRT3_12;
                let p2 = q2 + (3.0*q2 - 4.0*q1 + q0)*utils::SQRT3_12;

                q_l2[[i,j,k]] = w0*p0 + w1*p1 + w2*p2;

                let q0 = q_r[[i,j+1,k]];
                let q1 = q_r[[i,j+2,k]];
                let q2 = q_r[[i,j+3,k]];
                let q3 = q_r[[i,j+4,k]];
                let q4 = q_r[[i,j+5,k]];

                let beta0 = 13.0/12.0 * (q2 - 2.0*q3 + q4).powi(2)
                                + 0.25*(3.0*q2 - 4.0*q3 + q4).powi(2);
                let beta1 = 13.0/12.0*(q1 - 2.0*q2 + q3).powi(2)
                                + 0.25*(q1 - q3).powi(2);
                let beta2 = 13.0/12.0*(q0 - 2.0*q1 + q2).powi(2)
                                + 0.25*(q0 - 4.0*q1 + 3.0*q2).powi(2);

                let d0 = 0.1928406937;
                let d1 = 0.6111111111;
                let d2 = 0.1960481952;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = q2 + (3.0*q2 - 4.0*q3 + q4)*utils::SQRT3_12;
                let p1 = q2 - (-q1 + q3)*utils::SQRT3_12;
                let p2 = q2 - (3.0*q2 - 4.0*q1 + q0)*utils::SQRT3_12;

                q_r1[[i,j,k]] = w0*p0 + w1*p1 + w2*p2;

                let d0 = 0.1960481952;
                let d1 = 0.6111111111;
                let d2 = 0.1928406937;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = q2 - (3.0*q2 - 4.0*q3 + q4)*utils::SQRT3_12;
                let p1 = q2 - (q1 - q3)*utils::SQRT3_12;
                let p2 = q2 + (3.0*q2 - 4.0*q1 + q0)*utils::SQRT3_12;

                q_r2[[i,j,k]] = w0*p0 + w1*p1 + w2*p2;

                
            }
        }
    }
    let flux_l1 = utils::flux_x_local(q_l1.view(), gamma);
    let flux_l2 = utils::flux_x_local(q_l2.view(), gamma);
    let flux_r1 = utils::flux_x_local(q_r1.view(), gamma);
    let flux_r2 = utils::flux_x_local(q_r2.view(), gamma);

    let flux_l = 0.5*(flux_l1 + flux_l2);
    let flux_r = 0.5*(flux_r2 + flux_r1);
    let q_l = 0.5*(q_l1 + q_l2);
    let q_r = 0.5*(q_r1 + q_r2);
    (q_l, flux_l, q_r, flux_r)

}

pub(crate) fn weno_y_reconstruct_local_cons(u: ArrayView3<'_, f64>, gamma: f64)
-> (Array3<f64>,Array3<f64>,Array3<f64>,Array3<f64>)
{
    let (m,n,c) = u.dim();
    assert!(c>= 4, "u must have at least 4 channels");
    let mut u_con = Array3::<f64>::zeros((m,n,4));
    for i in 0..m {
        for j in 0..n {
            for k in 0..4 {
                u_con[[i,j,k]] = u[[i,j,k]];
            }
        }
    }

    let out_m = m;
    let out_n = n-5;

    let mut q_l = Array3::<f64>::zeros((out_m, out_n, 4));
    let mut q_r = Array3::<f64>::zeros((out_m, out_n, 4)); 

    for i in 0..out_m {
        for j in 0..out_n {
            for k in 0..4 {
                let u0 = u_con[[i,j,k]];
                let u1 = u_con[[i,j+1,k]];
                let u2 = u_con[[i,j+2,k]];
                let u3 = u_con[[i,j+3,k]];
                let u4 = u_con[[i,j+4,k]];
                let u5 = u_con[[i,j+5,k]];

                let beta0 = 13.0/12.0 * (u2 - 2.0*u3 + u4).powi(2)
                                + 0.25*(3.0*u2 - 4.0*u3 + u4).powi(2);
                let beta1 = 13.0/12.0*(u1 - 2.0*u2 + u3).powi(2)
                                + 0.25*(u1 - u3).powi(2);
                let beta2 = 13.0/12.0*(u0 - 2.0*u1 + u2).powi(2)
                                + 0.25*(u0 - 4.0*u1 + 3.0*u2).powi(2);
                let d0 = 0.3;
                let d1 = 0.6;
                let d2 = 0.1;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = - u4 + 5.0*u3 + 2.0*u2;
                let p1 = - u1 + 5.0*u2 + 2.0*u3;
                let p2 = 2.0*u0 - 7.0*u1 + 11.0*u2;

                q_l[[i,j,k]] = (w0*p0 + w1*p1 + w2*p2)/6.0;

                let beta0 = 13.0/12.0 * (u3 - 2.0*u4 + u5).powi(2)
                                + 0.25*(3.0*u3 - 4.0*u4 + u5).powi(2);
                let beta1 = 13.0/12.0*(u2 - 2.0*u3 + u4).powi(2)
                                + 0.25*(u2 - u4).powi(2);
                let beta2 = 13.0/12.0*(u1 - 2.0*u2 + u3).powi(2)
                                + 0.25*(u1 - 4.0*u2 + 3.0*u3).powi(2);
                let d0 = 0.1;
                let d1 = 0.6;
                let d2 = 0.3;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = 2.0*u5 - 7.0*u4 + 11.0*u3;
                let p1 = -u4 + 5.0*u3 + 2.0*u2;
                let p2 = -u1 + 5.0*u2 + 2.0*u3;

                q_r[[i,j,k]] = (w0*p0+w1*p1+w2*p2)/6.0;
            }
        }
    }

    let m1 = out_m - 6;
    let n1 = out_n;

    let mut q_l1 = Array3::<f64>::zeros((m1,n1,4));
    let mut q_l2 = Array3::<f64>::zeros((m1,n1,4));
    let mut q_r1 = Array3::<f64>::zeros((m1,n1,4));
    let mut q_r2 = Array3::<f64>::zeros((m1,n1,4));

    for i in 0..m1 {
        for j in 0..n1 {
            for k in 0..4 {
                let q0 = q_l[[i+1,j,k]];
                let q1 = q_l[[i+2,j,k]];
                let q2 = q_l[[i+3,j,k]];
                let q3 = q_l[[i+4,j,k]];
                let q4 = q_l[[i+5,j,k]];

                let beta0 = 13.0/12.0 * (q2 - 2.0*q3 + q4).powi(2)
                                + 0.25*(3.0*q2 - 4.0*q3 + q4).powi(2);
                let beta1 = 13.0/12.0*(q1 - 2.0*q2 + q3).powi(2)
                                + 0.25*(q1 - q3).powi(2);
                let beta2 = 13.0/12.0*(q0 - 2.0*q1 + q2).powi(2)
                                + 0.25*(q0 - 4.0*q1 + 3.0*q2).powi(2);

                let d0 = 0.1928406937;
                let d1 = 0.6111111111;
                let d2 = 0.1960481952;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = q2 + (3.0*q2 - 4.0*q3 + q4)*utils::SQRT3_12;
                let p1 = q2 - (-q1 + q3)*utils::SQRT3_12;
                let p2 = q2 - (3.0*q2 - 4.0*q1 + q0)*utils::SQRT3_12;

                q_l1[[i,j,k]] = w0*p0 + w1*p1 + w2*p2;

                let d0 = 0.1960481952;
                let d1 = 0.6111111111;
                let d2 = 0.1928406937;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = q2 - (3.0*q2 - 4.0*q3 + q4)*utils::SQRT3_12;
                let p1 = q2 - (q1 - q3)*utils::SQRT3_12;
                let p2 = q2 + (3.0*q2 - 4.0*q1 + q0)*utils::SQRT3_12;

                q_l2[[i,j,k]] = w0*p0 + w1*p1 + w2*p2;

                let q0 = q_r[[i+1,j,k]];
                let q1 = q_r[[i+2,j,k]];
                let q2 = q_r[[i+3,j,k]];
                let q3 = q_r[[i+4,j,k]];
                let q4 = q_r[[i+5,j,k]];

                let beta0 = 13.0/12.0 * (q2 - 2.0*q3 + q4).powi(2)
                                + 0.25*(3.0*q2 - 4.0*q3 + q4).powi(2);
                let beta1 = 13.0/12.0*(q1 - 2.0*q2 + q3).powi(2)
                                + 0.25*(q1 - q3).powi(2);
                let beta2 = 13.0/12.0*(q0 - 2.0*q1 + q2).powi(2)
                                + 0.25*(q0 - 4.0*q1 + 3.0*q2).powi(2);

                let d0 = 0.1928406937;
                let d1 = 0.6111111111;
                let d2 = 0.1960481952;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = q2 + (3.0*q2 - 4.0*q3 + q4)*utils::SQRT3_12;
                let p1 = q2 - (-q1 + q3)*utils::SQRT3_12;
                let p2 = q2 - (3.0*q2 - 4.0*q1 + q0)*utils::SQRT3_12;

                q_r1[[i,j,k]] = w0*p0 + w1*p1 + w2*p2;

                let d0 = 0.1960481952;
                let d1 = 0.6111111111;
                let d2 = 0.1928406937;

                let a0 = d0/(utils::DEFAULT_EPS + beta0).powi(2);
                let a1 = d1/(utils::DEFAULT_EPS + beta1).powi(2);
                let a2 = d2/(utils::DEFAULT_EPS + beta2).powi(2);

                let sum_o = a0 + a1 + a2;

                let w0 = a0/sum_o;
                let w1 = a1/sum_o;
                let w2 = a2/sum_o;

                let p0 = q2 - (3.0*q2 - 4.0*q3 + q4)*utils::SQRT3_12;
                let p1 = q2 - (q1 - q3)*utils::SQRT3_12;
                let p2 = q2 + (3.0*q2 - 4.0*q1 + q0)*utils::SQRT3_12;

                q_r2[[i,j,k]] = w0*p0 + w1*p1 + w2*p2;

                
            }
        }
    }

    let flux_l1 = utils::flux_y_local(q_l1.view(), gamma);
    let flux_l2 = utils::flux_y_local(q_l2.view(), gamma);
    let flux_r1 = utils::flux_y_local(q_r1.view(), gamma);
    let flux_r2 = utils::flux_y_local(q_r2.view(), gamma);

    let flux_l = 0.5*(flux_l1 + flux_l2);
    let flux_r = 0.5*(flux_r2 + flux_r1);
    let q_l = 0.5*(q_l1 + q_l2);
    let q_r = 0.5*(q_r1 + q_r2);
    (q_l, flux_l, q_r, flux_r)
}