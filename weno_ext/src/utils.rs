use ndarray::{Array2, Array3, ArrayView3};

pub(crate) const DEFAULT_EPS: f64 = 1e-12_f64;
pub(crate) const SQRT3_12: f64 = 0.1443375673;

/// conservative -> primitive; returns 2D fields (Array2)
pub(crate) fn con2primi_local(q: ArrayView3<'_, f64>, gamma: f64)
    -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>)
{
    let (m, n, _c) = q.dim();
    let mut p = Array2::<f64>::zeros((m, n));
    let mut a = Array2::<f64>::zeros((m, n));
    let mut rho_safe = Array2::<f64>::zeros((m, n));
    let mut u = Array2::<f64>::zeros((m, n));
    let mut v = Array2::<f64>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let rho = q[[i, j, 0]].max(DEFAULT_EPS);
            rho_safe[[i, j]] = rho;
            let uval = q[[i, j, 1]] / rho;
            let vval = q[[i, j, 2]] / rho;
            u[[i, j]] = uval;
            v[[i, j]] = vval;
            let pval = ((gamma - 1.0) * (q[[i, j, 3]] - 0.5 * rho * (uval*uval + vval*vval))).max(DEFAULT_EPS);
            p[[i, j]] = pval;
            a[[i, j]] = (gamma * pval / rho).sqrt();
        }
    }
    (p, a, rho_safe, u, v)
}

pub(crate) fn flux_x_local(q: ArrayView3<'_, f64>, gamma: f64) -> Array3<f64> {
    let (m, n, _c) = q.dim();
    let mut f = Array3::<f64>::zeros((m, n, 4));
    for i in 0..m {
        for j in 0..n {
            let rho = q[[i, j, 0]].max(DEFAULT_EPS);
            let u = q[[i, j, 1]] / rho;
            let v = q[[i, j, 2]] / rho;
            let p = (gamma - 1.0) * (q[[i, j, 3]] - 0.5 * rho * (u * u + v * v));
            f[[i, j, 0]] = q[[i, j, 1]];
            f[[i, j, 1]] = q[[i, j, 1]] * u + p;
            f[[i, j, 2]] = q[[i, j, 2]] * u;
            f[[i, j, 3]] = u * (q[[i, j, 3]] + p);
        }
    }
    f
}

/// flux_y - same but y-normal
pub(crate) fn flux_y_local(q: ArrayView3<'_, f64>, gamma: f64) -> Array3<f64> {
    let (m, n, _c) = q.dim();
    let mut f = Array3::<f64>::zeros((m, n, 4));
    for i in 0..m {
        for j in 0..n {
            let rho = q[[i, j, 0]].max(DEFAULT_EPS);
            let u = q[[i, j, 1]] / rho;
            let v = q[[i, j, 2]] / rho;
            let p = (gamma - 1.0) * (q[[i, j, 3]] - 0.5 * rho * (u * u + v * v));
            f[[i, j, 0]] = q[[i, j, 2]];
            f[[i, j, 1]] = q[[i, j, 1]] * v;
            f[[i, j, 2]] = q[[i, j, 2]] * v + p;
            f[[i, j, 3]] = v * (q[[i, j, 3]] + p);
        }
    }
    f
}