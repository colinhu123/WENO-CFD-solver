use ndarray::{ArrayView3, Array3};
use crate::utils;

pub(crate) fn weno_x_reconstruct_local(u: ArrayView3<'_, f64>, gamma: f64) -> 
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

pub(crate) fn weno_y_reconstruct_local(u: ArrayView3<'_, f64>, gamma: f64)
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