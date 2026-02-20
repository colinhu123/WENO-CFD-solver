// src/lib.rs
use pyo3::prelude::*;
use numpy::{PyArray3, PyReadonlyArray3, IntoPyArray};
use ndarray::{s, Array3};
use std::f64;

const SQRT3_12: f64 = 0.1443375673;

#[pyfunction]
fn hllc_x_rs(
    py: Python,
    qL: PyReadonlyArray3<f64>, // shape (M,N,4)
    qR: PyReadonlyArray3<f64>,
    F_L: PyReadonlyArray3<f64>,
    F_R: PyReadonlyArray3<f64>,
    gamma: f64,
) -> PyResult<Py<PyArray3<f64>>> {
    let ql = qL.as_array();
    let qr = qR.as_array();
    let fl = F_L.as_array();
    let fr = F_R.as_array();

    // shape checks
    if ql.shape() != qr.shape() || ql.shape() != fl.shape() || ql.shape() != fr.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input arrays must have identical shapes",
        ));
    }
    if ql.ndim() != 3 || ql.shape()[2] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected input shape (M,N,4)",
        ));
    }

    let (m, n, _c) = (ql.shape()[0], ql.shape()[1], ql.shape()[2]);
    let mut out = Array3::<f64>::zeros((m, n, 4));

    let eps = 1e-12_f64;

    // iterate over faces (i,j)
    for i in 0..m {
        for j in 0..n {
            // fetch left / right conserved variables
            let ql_ij = ql.slice(s![i, j, ..]);
            let qr_ij = qr.slice(s![i, j, ..]);
            let fl_ij = fl.slice(s![i, j, ..]);
            let fr_ij = fr.slice(s![i, j, ..]);

            // conservative -> primitive (per cell)
            // q[...,0]=rho, q[...,1]=rho*u, q[...,2]=rho*v, q[...,3]=E
            let rhoL = (ql_ij[0]).max(eps);
            let rhoR = (qr_ij[0]).max(eps);
            let uL = ql_ij[1] / rhoL;
            let vL = ql_ij[2] / rhoL;
            let uR = qr_ij[1] / rhoR;
            let vR = qr_ij[2] / rhoR;
            let pL = (gamma - 1.0) * (ql_ij[3] - 0.5 * rhoL * (uL * uL + vL * vL));
            let pR = (gamma - 1.0) * (qr_ij[3] - 0.5 * rhoR * (uR * uR + vR * vR));
            let pL = pL.max(eps);
            let pR = pR.max(eps);

            let HL = (ql_ij[3] + pL) / rhoL;
            let HR = (qr_ij[3] + pR) / rhoR;

            let sqrt_rhoL = rhoL.sqrt();
            let sqrt_rhoR = rhoR.sqrt();
            let denom_r = sqrt_rhoL + sqrt_rhoR;
            // Roe-like tilde states
            let u_tilde = (sqrt_rhoL * uL + sqrt_rhoR * uR) / denom_r;
            let v_tilde = (sqrt_rhoL * vL + sqrt_rhoR * vR) / denom_r;
            let H_tilde = (sqrt_rhoL * HL + sqrt_rhoR * HR) / denom_r;
            let tmp = H_tilde - 0.5 * (u_tilde * u_tilde + v_tilde * v_tilde);
            let a_tilde = ((gamma - 1.0) * tmp).max(0.0).sqrt();

            let SL = u_tilde - a_tilde;
            let SR = u_tilde + a_tilde;

            // denom for Sstar (avoid division by zero)
            let mut denom = rhoL * (SL - uL) - rhoR * (SR - uR);
            if denom.abs() < eps {
                denom = eps.copysign(denom) + eps;
            }

            // S*
            let Sstar = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / denom;

            // Left star conserved
            let coeffL = rhoL * (SL - uL) / (SL - Sstar);
            // specific energy
            let eL_spec = ql_ij[3] / rhoL;
            let Estar_spec_L = eL_spec + (Sstar - uL) * (Sstar + pL / (rhoL * (SL - uL)));
            let mut QstarL = [0.0_f64; 4];
            QstarL[0] = coeffL;
            QstarL[1] = coeffL * Sstar;
            QstarL[2] = coeffL * vL;
            QstarL[3] = coeffL * Estar_spec_L;

            // Right star conserved
            let coeffR = rhoR * (Sstar - uR) / (SR - Sstar);
            let eR_spec = qr_ij[3] / rhoR;
            let Estar_spec_R = eR_spec + (Sstar - uR) * (Sstar + pR / (rhoR * (SR - uR)));
            let mut QstarR = [0.0_f64; 4];
            QstarR[0] = coeffR;
            QstarR[1] = coeffR * Sstar;
            QstarR[2] = coeffR * vR;
            QstarR[3] = coeffR * Estar_spec_R;

            // Now pick flux according to wave speeds
            let mut Fij = [0.0_f64; 4];
            if SL >= 0.0 {
                // left-going everything
                for k in 0..4 {
                    Fij[k] = fl_ij[k];
                }
            } else if SL < 0.0 && Sstar >= 0.0 {
                // star-left region
                for k in 0..4 {
                    Fij[k] = fl_ij[k] + SL * (QstarL[k] - ql_ij[k]);
                }
            } else if Sstar < 0.0 && SR > 0.0 {
                // star-right region
                for k in 0..4 {
                    Fij[k] = fr_ij[k] + SR * (QstarR[k] - qr_ij[k]);
                }
            } else {
                // SR <= 0 : right-going everything
                for k in 0..4 {
                    Fij[k] = fr_ij[k];
                }
            }

            // write out
            for k in 0..4 {
                out[[i, j, k]] = Fij[k];
            }
        }
    }

    Ok(out.into_pyarray(py).to_owned())
}


#[pyfunction]
fn weno_x_rs(
    py: Python,
    q: PyReadonlyArray3<f64>,   // shape (nx, ny, 4)
    eps: f64,
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray3<f64>>)> {

    let q = q.as_array();
    if q.ndim() != 3 || q.shape()[2] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected input shape (nx, ny, 4)",
        ));
    }

    let nx = q.shape()[0];
    let ny = q.shape()[1];
    let nv = 4;

    // Output: interfaces in x-direction
    let mut qL = Array3::<f64>::zeros((nx - 5, ny, nv));
    let mut qR = Array3::<f64>::zeros((nx - 5, ny, nv));

    let gamma = 1e-6;

    for j in 0..ny {
        for k in 0..nv {

            // loop over interior interfaces
            for i in 2..(nx - 3) {

                // stencil values
                let v0 = q[[i-2, j, k]];
                let v1 = q[[i-1, j, k]];
                let v2 = q[[i,   j, k]];
                let v3 = q[[i+1, j, k]];
                let v4 = q[[i+2, j, k]];
                let v5 = q[[i+3, j, k]];

                // ---- LEFT state at i+1/2 ----

                let p0 = (2.0*v0 - 7.0*v1 + 11.0*v2) / 6.0;
                let p1 = (-v1 + 5.0*v2 + 2.0*v3) / 6.0;
                let p2 = (2.0*v2 + 5.0*v3 - v4) / 6.0;

                let b0 = (13.0/12.0)*(v0 - 2.0*v1 + v2).powi(2)
                       + 0.25*(v0 - 4.0*v1 + 3.0*v2).powi(2);
                let b1 = (13.0/12.0)*(v1 - 2.0*v2 + v3).powi(2)
                       + 0.25*(v1 - v3).powi(2);
                let b2 = (13.0/12.0)*(v2 - 2.0*v3 + v4).powi(2)
                       + 0.25*(3.0*v2 - 4.0*v3 + v4).powi(2);

                let a0 = 0.1 / ( (eps + b0).powi(2) );
                let a1 = 0.6 / ( (eps + b1).powi(2) );
                let a2 = 0.3 / ( (eps + b2).powi(2) );

                let asum = a0 + a1 + a2;

                let w0 = a0 / asum;
                let w1 = a1 / asum;
                let w2 = a2 / asum;

                qL[[i+1, j, k]] = w0*p0 + w1*p1 + w2*p2;



                // ---- RIGHT state (mirror stencil) ----

                let p0r = (-v1 + 5.0*v2 + 2.0*v3) / 6.0;
                let p1r = (2.0*v2 + 5.0*v3 - v4) / 6.0;
                let p2r = (11.0*v3 - 7.0*v4 + 2.0*v5) / 6.0;

                let b0r = (13.0/12.0)*(v1 - 2.0*v2 + v3).powi(2)
                        + 0.25*(v1 - 4.0*v2 + 3.0*v3).powi(2);
                let b1r = (13.0/12.0)*(v2 - 2.0*v3 + v4).powi(2)
                        + 0.25*(v2 - v4).powi(2);
                let b2r = (13.0/12.0)*(v3 - 2.0*v4 + v5).powi(2)
                        + 0.25*(3.0*v3 - 4.0*v4 + v5).powi(2);

                let a0r = 0.3 / ( (eps + b0r).powi(2) );
                let a1r = 0.6 / ( (eps + b1r).powi(2) );
                let a2r = 0.1 / ( (eps + b2r).powi(2) );

                let asumr = a0r + a1r + a2r;

                let w0r = a0r / asumr;
                let w1r = a1r / asumr;
                let w2r = a2r / asumr;

                qR[[i+1, j, k]] = w0r*p0r + w1r*p1r + w2r*p2r;
            }
        }
    }

    Ok((
        qL.into_pyarray(py).to_owned(),
        qR.into_pyarray(py).to_owned()
    ))
}


#[pymodule]
fn weno_ext(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hllc_x_rs, m)?)?;
    m.add_function(wrap_pyfunction!(weno_x_rs, m)?)?;
    Ok(())
}
