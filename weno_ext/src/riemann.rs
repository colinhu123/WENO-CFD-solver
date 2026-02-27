use ndarray::{ArrayView3,Array3};
use crate::utils;



pub(crate) fn hllc_x_local(q_l: ArrayView3<'_,f64>,
                            f_l:ArrayView3<'_,f64>,
                            q_r: ArrayView3<'_,f64>,
                            f_r:ArrayView3<'_,f64>,
                            gamma:f64){

    let (m, n, _c) = q_l.dim();
    let mut out = Array3::<f64>::zeros((m, n, 4));

    let (p_l,a_l,rho_l,u_l,v_l, h_l) = utils::con2primi_local(q_l, gamma);
    let (p_r,a_r,rho_r,u_r,v_r, h_r) = utils::con2primi_local(q_r, gamma);

    let u_tilde = utils::roe_average(u_l.view(), u_r.view(), rho_l.view(), rho_r.view());
    let v_tilde = utils::roe_average(v_l.view(), v_r.view(), rho_l.view(), rho_r.view());
    let h_tilde = utils::roe_average(h_l.view(), h_r.view(), rho_l.view(), rho_r.view());

    let a_tilde = utils::a_calc(u_tilde.view(), v_tilde.view(), h_tilde.view(), gamma);

    
}