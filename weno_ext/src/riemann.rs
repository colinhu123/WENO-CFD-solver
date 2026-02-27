use ndarray::{ArrayView3,Array3};
use crate::utils;



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
{

}