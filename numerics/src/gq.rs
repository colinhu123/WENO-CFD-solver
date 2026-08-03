use physics::{State};

pub struct gqStencil {
    points: [State; 5],
}

pub fn gq(u0: f64, u1: f64, u2: f64, u3: f64, u4: f64) -> (f64, f64, f64) {
    (0.0, 0.0, 0.0)
}