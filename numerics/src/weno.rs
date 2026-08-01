use crate::utils;

use physics;

pub struct WenoStencil {
    pub points: [physics::State; 6],
}

pub fn weno5(u0: f64,u1: f64,u2: f64,u3: f64,u4: f64) -> f64 {
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

    (w0*p0 + w1*p1 + w2*p2)/6.0
}

pub fn weno_recon(stencil: [f64; 6]) -> (f64, f64){
    let u_plus = weno5(stencil[0],stencil[1],stencil[2],stencil[3],stencil[4]);
    let u_minus = weno5(stencil[5],stencil[4],stencil[3],stencil[2],stencil[1]);

    (u_plus,u_minus)
}

impl WenoStencil {
    pub fn state2arr(&self) -> [[f64;6];4] {
        let points = self.points;
        [
            [points[0].rho,points[1].rho,points[2].rho,points[3].rho,points[4].rho,points[5].rho],
            [points[0].mom_x,points[1].mom_x,points[2].mom_x,points[3].mom_x,points[4].mom_x,points[5].mom_x],
            [points[0].mom_y,points[1].mom_y,points[2].mom_y,points[3].mom_y,points[4].mom_y,points[5].mom_y],
            [points[0].e,points[1].e,points[2].e,points[3].e,points[4].e,points[5].e]
        ]
    }


    ///directly weno reconstruction, without integrate charateristic projection
    /// suppose use with state.con2char & char2con
    /// Output: (left state, right state)
    pub fn weno_reconstruction(&self) -> (physics::State,physics::State) {
        let tmp = self.state2arr();

        let (rho_l, rho_r) = weno_recon(tmp[0]);
        let (mom_xl, mom_xr) = weno_recon(tmp[1]);
        let (mom_yl, mom_yr) = weno_recon(tmp[2]);
        let (el, er) = weno_recon(tmp[3]);

        let state_l = physics::State {
            rho: rho_l,
            mom_x: mom_xl,
            mom_y: mom_yl,
            e: el,
        };
        let state_r = physics::State {
            rho: rho_r,
            mom_x: mom_xr,
            mom_y: mom_yr,
            e: er,
        };
        (state_l, state_r)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn test_weno5_constant() {
        let u = [3.14; 5];

        let result = weno5(
            u[0], u[1], u[2], u[3], u[4]
        );

        assert!(
            (result - 3.14).abs() < TOL,
            "constant reconstruction failed: {}",
            result
        );
    }


    #[test]
    fn test_weno5_linear_function() {
        // u(x)=2x+1
        //
        // stencil:
        // x=-2,-1,0,1,2
        //
        // value at interface x=0.5 is 2
        let result = weno5(
            -3.0,
            -1.0,
             1.0,
             3.0,
             5.0
        );

        let exact = 2.0;

        assert!(
            (result - exact).abs() < 1e-12,
            "linear reconstruction inaccurate: {}, expected {}",
            result,
            exact
        );
    }


    #[test]
    fn test_weno5_quadratic_function() {
        // u=x^2
        //
        // stencil:
        // -2,-1,0,1,2
        //
        // interface x=0.5 -> 0.25

        let result = weno5(
            4.0 + 1.0/12.0,
            1.0 + 1.0/12.0,
            0.0 + 1.0/12.0,
            1.0 + 1.0/12.0,
            4.0 + 1.0/12.0,
        );

        let exact = 0.25;

        assert!(
            (result - exact).abs() < 1e-12,
            "quadratic reconstruction inaccurate: {}, expected {}",
            result,
            exact
        );
    }


    #[test]
    fn test_weno_recon_symmetry() {
        let stencil = [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ];

        let (left, right) = weno_recon(stencil);

        let (mirror_right, mirror_left) = weno_recon([
            6.0,
            5.0,
            4.0,
            3.0,
            2.0,
            1.0
        ]);

        assert!(
            (left - mirror_right).abs() < TOL,
            "left/right symmetry broken"
        );

        assert!(
            (right - mirror_left).abs() < TOL,
            "left/right symmetry broken"
        );
    }


    #[test]
    fn test_weno_recon_constant() {
        let stencil = [5.0; 6];

        let (left, right) = weno_recon(stencil);

        assert!(
            (left - 5.0).abs() < TOL
        );

        assert!(
            (right - 5.0).abs() < TOL
        );
    }


    #[test]
    fn test_weno_reconstruction_state() {
        let points = [
            physics::State {
                rho: 1.0,
                mom_x: 2.0,
                mom_y: 3.0,
                e: 4.0,
            };
            6
        ];

        let stencil = WenoStencil {
            points
        };

        let (left,right) = stencil.weno_reconstruction();


        assert!((left.rho - 1.0).abs() < TOL);
        assert!((left.mom_x - 2.0).abs() < TOL);
        assert!((left.mom_y - 3.0).abs() < TOL);
        assert!((left.e - 4.0).abs() < TOL);


        assert!((right.rho - 1.0).abs() < TOL);
        assert!((right.mom_x - 2.0).abs() < TOL);
        assert!((right.mom_y - 3.0).abs() < TOL);
        assert!((right.e - 4.0).abs() < TOL);
    }


    #[test]
    fn test_weno5_no_overshoot_at_jump() {
        // discontinuity:
        //
        // [0,0,0,1,1]
        //
        // WENO should not create large oscillations

        let result = weno5(
            0.0,
            0.0,
            0.0,
            1.0,
            1.0
        );

        assert!(
            result >= -1e-12 && result <= 1.0 + 1e-12,
            "WENO produced oscillation: {}",
            result
        );
    }
}