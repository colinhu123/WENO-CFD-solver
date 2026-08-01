use physics;

pub struct GodunovStencil {
    pub points: [physics::State; 2],
}

pub fn godunov(stencil: [f64;2])-> (f64, f64){
    (stencil[0],stencil[1])
}

impl GodunovStencil {

    pub fn state2arr(&self)->[[f64;2];4] {
        let points = self.points;
        [
            [points[0].rho,points[1].rho],
            [points[0].mom_x, points[1].mom_x],
            [points[0].mom_y, points[1].mom_y],
            [points[0].e, points[1].e],
        ]
    }

    pub fn godunov_reconstruction(&self) -> (physics::State, physics::State) {

        let tmp = self.state2arr();

        let (rho_l, rho_r) = godunov(tmp[0]);
        let (mom_xl, mom_xr) = godunov(tmp[1]);
        let (mom_yl, mom_yr) = godunov(tmp[2]);
        let (el, er) = godunov(tmp[3]);

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