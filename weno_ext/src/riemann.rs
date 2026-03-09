use ndarray::{ArrayView3,Array3, ArrayView2, Array2};
use crate::utils;
use crate::weno;


pub(crate) fn hllc_x_local(q_l: ArrayView3<'_,f64>,
                            f_l:ArrayView3<'_,f64>,
                            q_r: ArrayView3<'_,f64>,
                            f_r:ArrayView3<'_,f64>,
                            gamma:f64)
                            -> Array3<f64>{

    let (m, n, _c) = q_l.dim();
    let mut out = Array3::<f64>::zeros((m, n, 4));
    

    for i in 0..m {
        for j in 0..n {
            // load
            let ql0 = q_l[[i, j, 0]].max(utils::DEFAULT_EPS);
            let ql1 = q_l[[i, j, 1]];
            let ql2 = q_l[[i, j, 2]];
            let ql3 = q_l[[i, j, 3]];
            let qr0 = q_r[[i, j, 0]].max(utils::DEFAULT_EPS);
            let qr1 = q_r[[i, j, 1]];
            let qr2 = q_r[[i, j, 2]];
            let qr3 = q_r[[i, j, 3]];

            let u_l = ql1 / ql0;
            let v_l = ql2 / ql0;
            let u_r = qr1 / qr0;
            let v_r = qr2 / qr0;
            let p_l = (gamma - 1.0) * (ql3 - 0.5 * ql0 * (u_l * u_l + v_l * v_l));
            let p_r = (gamma - 1.0) * (qr3 - 0.5 * qr0 * (u_r * u_r + v_r * v_r));
            let p_l = p_l.max(utils::DEFAULT_EPS);
            let p_r = p_r.max(utils::DEFAULT_EPS);
            let h_l = (ql3 + p_l) / ql0;
            let h_r = (qr3 + p_r) / qr0;

            let sqrt_rho_l = ql0.sqrt();
            let sqrt_rho_r = qr0.sqrt();    
            let denom_r = sqrt_rho_l + sqrt_rho_r;
            let u_tilde = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) / denom_r;
            let v_tilde = (sqrt_rho_l * v_l + sqrt_rho_r * v_r) / denom_r;
            let h_tilde = (sqrt_rho_l * h_l + sqrt_rho_r * h_r) / denom_r;
            let tmp = h_tilde - 0.5 * (u_tilde * u_tilde + v_tilde * v_tilde);
            let a_tilde = ((gamma - 1.0) * tmp).max(0.0).sqrt();

            let s_l = u_tilde - a_tilde;
            let s_r = u_tilde + a_tilde;

            // denom for Sstar
            let mut denom = ql0 * (s_l - u_l) - qr0 * (s_r - u_r);
            if denom.abs() < utils::DEFAULT_EPS {
                denom = utils::DEFAULT_EPS.copysign(denom) + utils::DEFAULT_EPS;
            }

            let s_star = (p_r - p_l + ql0 * u_l * (s_l - u_l) - qr0 * u_r * (s_r - u_r)) / denom;

            // Left star
            let coeff_l = ql0 * (s_l - u_l) / (s_l - s_star);
            let el_spec = ql3 / ql0;
            let e_star_spec_l = el_spec + (s_star - u_l) * (s_star + p_l / (ql0 * (s_l - u_l)));
            let q_star_l = [coeff_l, coeff_l * s_star, coeff_l * v_l, coeff_l * e_star_spec_l];

            // Right star
            let coeff_r = qr0 * (s_star - u_r) / (s_r - s_star);
            let er_spec = qr3 / qr0;
            let e_star_spec_r = er_spec + (s_star - u_r) * (s_star + p_r / (qr0 * (s_r - u_r)));
            let q_star_r = [coeff_r, coeff_r * s_star, coeff_r * v_r, coeff_r * e_star_spec_r];

            // pick flux
            let fl = [f_l[[i, j, 0]], f_l[[i, j, 1]], f_l[[i, j, 2]], f_l[[i, j, 3]]];
            let fr = [f_r[[i, j, 0]], f_r[[i, j, 1]], f_r[[i, j, 2]], f_r[[i, j, 3]]];

            let mut fij = [0.0_f64; 4];
            if s_l >= 0.0 {
                fij = fl;
            } else if s_l < 0.0 && s_star >= 0.0 {
                for k in 0..4 {
                    fij[k] = fl[k] + s_l * (q_star_l[k] - q_l[[i, j, k]]);
                }
            } else if s_star < 0.0 && s_r > 0.0 {
                for k in 0..4 {
                    fij[k] = fr[k] + s_r * (q_star_r[k] - q_r[[i, j, k]]);
                }
            } else {
                fij = fr;
            }
            for k in 0..4 {
                out[[i, j, k]] = fij[k];
            }
        }
    }
    out
}


pub(crate) fn hllc_y_local(q_l: ArrayView3<'_,f64>,
                            f_l:ArrayView3<'_,f64>,
                            q_r: ArrayView3<'_,f64>,
                            f_r:ArrayView3<'_,f64>,
                            gamma:f64)
                            -> Array3<f64>{

    let (m, n, _c) = q_l.dim();
    let mut out = Array3::<f64>::zeros((m, n, 4));
    

    for i in 0..m {
        for j in 0..n {
            // load
            let ql0 = q_l[[i, j, 0]].max(utils::DEFAULT_EPS);
            let ql1 = q_l[[i, j, 1]];
            let ql2 = q_l[[i, j, 2]];
            let ql3 = q_l[[i, j, 3]];
            let qr0 = q_r[[i, j, 0]].max(utils::DEFAULT_EPS);
            let qr1 = q_r[[i, j, 1]];
            let qr2 = q_r[[i, j, 2]];
            let qr3 = q_r[[i, j, 3]];

            let u_l = ql1 / ql0;
            let v_l = ql2 / ql0;
            let u_r = qr1 / qr0;
            let v_r = qr2 / qr0;
            let p_l = (gamma - 1.0) * (ql3 - 0.5 * ql0 * (u_l * u_l + v_l * v_l));
            let p_r = (gamma - 1.0) * (qr3 - 0.5 * qr0 * (u_r * u_r + v_r * v_r));
            let p_l = p_l.max(utils::DEFAULT_EPS);
            let p_r = p_r.max(utils::DEFAULT_EPS);
            let h_l = (ql3 + p_l) / ql0;
            let h_r = (qr3 + p_r) / qr0;

            let sqrt_rho_l = ql0.sqrt();
            let sqrt_rho_r = qr0.sqrt();    
            let denom_r = sqrt_rho_l + sqrt_rho_r;
            let u_tilde = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) / denom_r;
            let v_tilde = (sqrt_rho_l * v_l + sqrt_rho_r * v_r) / denom_r;
            let h_tilde = (sqrt_rho_l * h_l + sqrt_rho_r * h_r) / denom_r;
            let tmp = h_tilde - 0.5 * (u_tilde * u_tilde + v_tilde * v_tilde);
            let a_tilde = ((gamma - 1.0) * tmp).max(0.0).sqrt();

            let s_l = v_tilde - a_tilde;
            let s_r = v_tilde + a_tilde;

            // denom for Sstar
            let mut denom = ql0 * (s_l - v_l) - qr0 * (s_r - v_r);
            if denom.abs() < utils::DEFAULT_EPS {
                denom = utils::DEFAULT_EPS.copysign(denom) + utils::DEFAULT_EPS;
            }

            let s_star = (p_r - p_l + ql0 * v_l * (s_l - v_l) - qr0 * v_r * (s_r - v_r)) / denom;

            // Left star
            let coeff_l = ql0 * (s_l - v_l) / (s_l - s_star);
            let el_spec = ql3 / ql0;
            let e_star_spec_l = el_spec + (s_star - v_l) * (s_star + p_l / (ql0 * (s_l - v_l)));
            let q_star_l = [coeff_l, coeff_l * s_star, coeff_l * v_l, coeff_l * e_star_spec_l];

            // Right star
            let coeff_r = qr0 * (s_star - v_r) / (s_r - s_star);
            let er_spec = qr3 / qr0;
            let e_star_spec_r = er_spec + (s_star - v_r) * (s_star + p_r / (qr0 * (s_r - v_r)));
            let q_star_r = [coeff_r, coeff_r * s_star, coeff_r * v_r, coeff_r * e_star_spec_r];

            // pick flux
            let fl = [f_l[[i, j, 0]], f_l[[i, j, 1]], f_l[[i, j, 2]], f_l[[i, j, 3]]];
            let fr = [f_r[[i, j, 0]], f_r[[i, j, 1]], f_r[[i, j, 2]], f_r[[i, j, 3]]];

            let mut fij = [0.0_f64; 4];
            if s_l >= 0.0 {
                fij = fl;
            } else if s_l < 0.0 && s_star >= 0.0 {
                for k in 0..4 {
                    fij[k] = fl[k] + s_l * (q_star_l[k] - q_l[[i, j, k]]);
                }
            } else if s_star < 0.0 && s_r > 0.0 {
                for k in 0..4 {
                    fij[k] = fr[k] + s_r * (q_star_r[k] - q_r[[i, j, k]]);
                }
            } else {
                fij = fr;
            }
            for k in 0..4 {
                out[[i, j, k]] = fij[k];
            }
        }
    }
    out
}

pub(crate) fn hlle_x_local(q_l: ArrayView3<'_,f64>,
                            f_l:ArrayView3<'_,f64>,
                            q_r: ArrayView3<'_,f64>,
                            f_r:ArrayView3<'_,f64>,
                            gamma:f64)
-> Array3<f64>{
let (m, n, _c) = q_l.dim();
    let mut out = Array3::<f64>::zeros((m, n, 4));
    

    for i in 0..m {
        for j in 0..n {
            // load
            let ql0 = q_l[[i, j, 0]].max(utils::DEFAULT_EPS);
            let ql1 = q_l[[i, j, 1]];
            let ql2 = q_l[[i, j, 2]];
            let ql3 = q_l[[i, j, 3]];
            let qr0 = q_r[[i, j, 0]].max(utils::DEFAULT_EPS);
            let qr1 = q_r[[i, j, 1]];
            let qr2 = q_r[[i, j, 2]];
            let qr3 = q_r[[i, j, 3]];

            let u_l = ql1 / ql0;
            let v_l = ql2 / ql0;
            let u_r = qr1 / qr0;
            let v_r = qr2 / qr0;
            let p_l = (gamma - 1.0) * (ql3 - 0.5 * ql0 * (u_l * u_l + v_l * v_l));
            let p_r = (gamma - 1.0) * (qr3 - 0.5 * qr0 * (u_r * u_r + v_r * v_r));
            let p_l = p_l.max(utils::DEFAULT_EPS);
            let p_r = p_r.max(utils::DEFAULT_EPS);
            let h_l = (ql3 + p_l) / ql0;
            let h_r = (qr3 + p_r) / qr0;

            let sqrt_rho_l = ql0.sqrt();
            let sqrt_rho_r = qr0.sqrt();    
            let denom_r = sqrt_rho_l + sqrt_rho_r;
            let u_tilde = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) / denom_r;
            let v_tilde = (sqrt_rho_l * v_l + sqrt_rho_r * v_r) / denom_r;
            let h_tilde = (sqrt_rho_l * h_l + sqrt_rho_r * h_r) / denom_r;
            let tmp = h_tilde - 0.5 * (u_tilde * u_tilde + v_tilde * v_tilde);
            let a_tilde = ((gamma - 1.0) * tmp).max(0.0).sqrt();

            let s_l = u_tilde - a_tilde;
            let s_r = u_tilde + a_tilde;

            // denom for Sstar
            let mut denom = s_r - s_l;
            if denom.abs() < utils::DEFAULT_EPS {
                denom = utils::DEFAULT_EPS.copysign(denom) + utils::DEFAULT_EPS;
            }


            // pick flux
            let fl = [f_l[[i, j, 0]], f_l[[i, j, 1]], f_l[[i, j, 2]], f_l[[i, j, 3]]];
            let fr = [f_r[[i, j, 0]], f_r[[i, j, 1]], f_r[[i, j, 2]], f_r[[i, j, 3]]];

            let mut fij = [0.0_f64; 4];

            if s_l >= 0.0 {
                fij = fl;
            } else if s_r < 0.0 {
                fij = fr;
            } else {
                //let denom = (s_r-s_l).max(utils::DEFAULT_EPS);
                let q_l = [ql0, ql1, ql2, ql3];
                let q_r = [qr0, qr1, qr2, qr3];
                for k in 0..4 {
                    fij[k] = s_r*fl[k] - s_l*fr[k] + s_r*s_l*(q_r[k] - q_l[k])/denom;
                }
                
            }
            for k in 0..4 {
                out[[i,j,k]] = fij[k];
            }
            
        }
    }
    out
}

pub(crate) fn hlle_y_local(q_l: ArrayView3<'_,f64>,
                            f_l:ArrayView3<'_,f64>,
                            q_r: ArrayView3<'_,f64>,
                            f_r:ArrayView3<'_,f64>,
                            gamma:f64)
-> Array3<f64>{
let (m, n, _c) = q_l.dim();
    let mut out = Array3::<f64>::zeros((m, n, 4));
    

    for i in 0..m {
        for j in 0..n {
            // load
            let ql0 = q_l[[i, j, 0]].max(utils::DEFAULT_EPS);
            let ql1 = q_l[[i, j, 1]];
            let ql2 = q_l[[i, j, 2]];
            let ql3 = q_l[[i, j, 3]];
            let qr0 = q_r[[i, j, 0]].max(utils::DEFAULT_EPS);
            let qr1 = q_r[[i, j, 1]];
            let qr2 = q_r[[i, j, 2]];
            let qr3 = q_r[[i, j, 3]];

            let u_l = ql1 / ql0;
            let v_l = ql2 / ql0;
            let u_r = qr1 / qr0;
            let v_r = qr2 / qr0;
            let p_l = (gamma - 1.0) * (ql3 - 0.5 * ql0 * (u_l * u_l + v_l * v_l));
            let p_r = (gamma - 1.0) * (qr3 - 0.5 * qr0 * (u_r * u_r + v_r * v_r));
            let p_l = p_l.max(utils::DEFAULT_EPS);
            let p_r = p_r.max(utils::DEFAULT_EPS);
            let h_l = (ql3 + p_l) / ql0;
            let h_r = (qr3 + p_r) / qr0;

            let sqrt_rho_l = ql0.sqrt();
            let sqrt_rho_r = qr0.sqrt();    
            let denom_r = sqrt_rho_l + sqrt_rho_r;
            let u_tilde = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) / denom_r;
            let v_tilde = (sqrt_rho_l * v_l + sqrt_rho_r * v_r) / denom_r;
            let h_tilde = (sqrt_rho_l * h_l + sqrt_rho_r * h_r) / denom_r;
            let tmp = h_tilde - 0.5 * (u_tilde * u_tilde + v_tilde * v_tilde);
            let a_tilde = ((gamma - 1.0) * tmp).max(0.0).sqrt();

            let s_l = v_tilde - a_tilde;
            let s_r = v_tilde + a_tilde;

            // denom for Sstar
            let mut denom = s_r - s_l;
            if denom.abs() < utils::DEFAULT_EPS {
                denom = utils::DEFAULT_EPS.copysign(denom) + utils::DEFAULT_EPS;
            }


            // pick flux
            let fl = [f_l[[i, j, 0]], f_l[[i, j, 1]], f_l[[i, j, 2]], f_l[[i, j, 3]]];
            let fr = [f_r[[i, j, 0]], f_r[[i, j, 1]], f_r[[i, j, 2]], f_r[[i, j, 3]]];

            let mut fij = [0.0_f64; 4];

            if s_l >= 0.0 {
                fij = fl;
            } else if s_r < 0.0 {
                fij = fr;
            } else {
                //let denom = (s_r-s_l).max(utils::DEFAULT_EPS);
                let q_l = [ql0, ql1, ql2, ql3];
                let q_r = [qr0, qr1, qr2, qr3];
                for k in 0..4 {
                    fij[k] = s_r*fl[k] - s_l*fr[k] + s_r*s_l*(q_r[k] - q_l[k])/denom;
                }
                
            }
            for k in 0..4 {
                out[[i,j,k]] = fij[k];
            }
            
        }
    }
    out
}


pub(crate) fn shock_sensor_grad_p(
    p: ArrayView2<'_, f64>,
    jp_low: f64,
    jp_high: f64,
) -> Array2<f64> {

    let (m, n) = p.dim();

    let min_dim = (m.min(n)).max(1) as f64;
    let dx = 1.0 / min_dim;

    let mut dpdx = Array2::<f64>::zeros((m, n));
    let mut dpdy = Array2::<f64>::zeros((m, n));

    // interior finite differences (central)
    for i in 1..(m-1) {
        for j in 0..n {
            dpdx[[i, j]] = (p[[i+1, j]] - p[[i-1, j]]) / (2.0 * dx);
        }
    }

    // x-boundaries
    for j in 0..n {
        dpdx[[0, j]] = (p[[1, j]] - p[[0, j]]) / (2.0 * dx);
        dpdx[[m-1, j]] = (p[[m-1, j]] - p[[m-2, j]]) / (2.0 * dx);
    }

    // y-derivatives
    for i in 0..m {
        for j in 1..(n-1) {
            dpdy[[i, j]] = (p[[i, j+1]] - p[[i, j-1]]) / (2.0 * dx);
        }

        dpdy[[i, 0]] = (p[[i, 1]] - p[[i, 0]]) / (2.0 * dx);
        dpdy[[i, n-1]] = (p[[i, n-1]] - p[[i, n-2]]) / (2.0 * dx);
    }

    // sensor
    let mut phi = Array2::<f64>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let grad = (dpdx[[i, j]].powi(2) + dpdy[[i, j]].powi(2)).sqrt();
            let jp = grad / p[[i, j]].max(utils::DEFAULT_EPS);

            let w = if jp <= jp_low {
                1.0
            } else if jp >= jp_high {
                0.0
            } else {
                1.0 - (jp - jp_low) / (jp_high - jp_low)
            };

            phi[[i, j]] = w;
        }
    }

    phi
}

pub(crate) fn hllc_hlle_blend_x_local(
    q_l: ArrayView3<'_,f64>,
    q_r: ArrayView3<'_,f64>,
    f_l: ArrayView3<'_,f64>,
    f_r:ArrayView3<'_,f64>,
    gamma:f64,
    jp_low: f64,
    jp_high: f64
)->Array3<f64> {
    let (p_l, _a_l, _rho_l, _u_l, _v_l, _h_l) = utils::con2primi_local(q_l, gamma);
    let (p_r, _a_r, _rho_r, _u_r, _v_r, _h_r) = utils::con2primi_local(q_r, gamma);
    
    let (n,m) = p_l.dim();
    let mut p = Array2::<f64>::zeros((n,m));
    for i in 0..n {
        for j in 0..m {
            p[[i,j]] = 0.5*(p_l[[i,j]]+p_r[[i,j]]);
        }
    }
    let phi = shock_sensor_grad_p(p.view(), jp_low, jp_high);

    let f_hllc = hllc_x_local(q_l, f_l, q_r, f_r, gamma);
    let f_hlle = hlle_x_local(q_l, f_l, q_r, f_r, gamma);
    let (m,n,_c) = f_hllc.dim();
    let mut flux = Array3::<f64>::zeros((m,n,4));

    for i in 0..m {
        for j in 0..n {
            let w = phi[[i,j]];
            for k in 0..4 {
                flux[[i,j,k]] = w* f_hllc[[i,j,k]] + (1.0 - w)*f_hlle[[i,j,k]];
            }
        }
    }

    flux
}

pub(crate) fn hllc_hlle_blend_y_local(
    q_l: ArrayView3<'_,f64>,
    q_r: ArrayView3<'_,f64>,
    f_l: ArrayView3<'_,f64>,
    f_r:ArrayView3<'_,f64>,
    gamma:f64,
    jp_low: f64,
    jp_high: f64
)->Array3<f64> {
    let (p_l, _a_l, _rho_l, _u_l, _v_l, _h_l) = utils::con2primi_local(q_l, gamma);
    let (p_r, _a_r, _rho_r, _u_r, _v_r, _h_r) = utils::con2primi_local(q_r, gamma);
    
    let (n,m) = p_l.dim();
    let mut p = Array2::<f64>::zeros((n,m));
    for i in 0..n {
        for j in 0..m {
            p[[i,j]] = 0.5*(p_l[[i,j]]+p_r[[i,j]]);
        }
    }
    let phi = shock_sensor_grad_p(p.view(), jp_low, jp_high);

    let f_hllc = hllc_y_local(q_l, f_l, q_r, f_r, gamma);
    let f_hlle = hlle_y_local(q_l, f_l, q_r, f_r, gamma);
    let (m,n,_c) = f_hllc.dim();
    let mut flux = Array3::<f64>::zeros((m,n,4));

    for i in 0..m {
        for j in 0..n {
            let w = phi[[i,j]];
            for k in 0..4 {
                flux[[i,j,k]] = w* f_hllc[[i,j,k]] + (1.0 - w)*f_hlle[[i,j,k]];
            }
        }
    }

    flux
}


pub(crate) fn l_local(u: ArrayView3<'_,f64>, dx:f64, gamma:f64, force_hlle: bool, jp_cri: (f64,f64))
-> Array3<f64>
{
    let (q_lx, f_lx, q_rx, f_rx) = weno::weno_x_reconstruct_local(u, gamma);
    let (q_ly, f_ly, q_ry, f_ry) = weno::weno_y_reconstruct_local(u, gamma);

    let (jp_low, jp_high) = jp_cri;

    let f_x = if force_hlle {
        hlle_x_local(q_lx.view(), f_lx.view(), q_rx.view(), f_rx.view(), gamma)
    } else {
        hllc_hlle_blend_x_local(q_lx.view(), q_rx.view(), f_lx.view(), f_rx.view(), gamma, jp_low, jp_high)
    };

    let f_y = if force_hlle {
        hlle_y_local(q_ly.view(), f_ly.view(), q_ry.view(), f_ry.view(), gamma)
    } else {
        hllc_hlle_blend_y_local(q_ly.view(), q_ry.view(), f_ly.view(), f_ry.view(), gamma, jp_low, jp_high)
    };

    let (mx, nx, _c) = f_x.dim();
    let mut df_dx = Array3::<f64>::zeros((mx-1, nx, 4));
    for i in 0..(mx-1) {
        for j in 0..nx {
            for k in 0..4 {
                df_dx[[i,j,k]] = (f_x[[i+1,j,k]] - f_x[[i,j,k]]) / dx;
            }
        }
    }
    let (my, ny, _c) = f_y.dim();
    let mut dg_dy = Array3::<f64>::zeros((my, ny-1, 4));
    for i in 0..my {
        for j in 0..(ny-1) {
            for k in 0..4 {
                dg_dy[[i,j,k]] = (f_y[[i,j+1,k]] - f_y[[i,j,k]]) / dx;
            }
        }
    }
    // combine into domain (mx-1, ny-1, 4) -> same as interior of u
    let mut out = Array3::<f64>::zeros((mx-1, ny-1, 4));
    for i in 0..(mx-1) {
        for j in 0..(ny-1) {
            for k in 0..4 {
                out[[i,j,k]] = -(df_dx[[i,j,k]] + dg_dy[[i,j,k]]);
            }
        }
    }
    out
}