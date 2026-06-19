/*
This file is defining the physics of the Euler state here.
input are expected as [f64; 4] while each terms is defined as the following
rho, rho*u, rho*v, e

pressure and density floor is also applied in this file.
*/

pub const RHO_MIN: f64 = 1e-12; // your existing density floor
pub const P_MIN:   f64 = 1e-12;
pub(crate) struct State4  {
    density: f64,
    mom_x: f64,
    mom_y: f64,
    e: f64
}


impl State4 {

    pub fn apply_density_floor(&mut self) {
        self.density = self.density.max(RHO_MIN);
    }

    pub fn pressure(&self,gamma: f64) -> f64 {
        if self.density <= 0.0 {return 0.0;} 
        else{
            let pres = (gamma - 1.0) * (self.e - 0.5 * (self.mom_x*self.mom_x + self.mom_y*self.mom_y) / self.density);
            pres
        }
    }

    pub fn sound_speed(&self, gamma: f64) -> f64 {
        let primi_vars = conserved_to_primitive(&self, gamma);
        let c = (gamma*primi_vars[3]/self.density).sqrt();
        c
    }

    pub fn total_enthalpy(&self, gamma: f64) -> f64 {
        (self.e + self.pressure(gamma))/self.density
    }
}

pub(crate) fn conserved_to_primitive(q: &State4,gamma: f64) -> [f64; 4] {
    let density = q.density;
    let velo_x = q.mom_x/q.density;
    let velo_y = q.mom_y/q.density;
    let pressure = q.pressure(gamma);
    [density, velo_x, velo_y, pressure]
}

pub(crate) fn primitive_to_conserved(rho: f64, u: f64, v: f64, pressure: f64, gamma: f64)-> State4{
    let mom_x = rho*u;
    let mom_y = rho*v;
    let e = pressure/(gamma - 1.0) + 0.5*rho*(v.powi(2)+u.powi(2));

    let mut q = State4 {
        density: rho,
        mom_x: mom_x,
        mom_y: mom_y,
        e: e
    };
    q.apply_density_floor();
    q
}

pub(crate) fn flux_x(q: &State4, gamma: f64) -> [f64; 4] {
    let primi_state = conserved_to_primitive(&q, gamma);
    let mass_flux = q.mom_x;
    let flux2 = q.mom_x*primi_state[1] + primi_state[3];
    let flux3 = q.mom_y*primi_state[1];
    let flux4 = primi_state[1]*(q.e + primi_state[3]);
    [mass_flux,flux2,flux3,flux4]
}

pub(crate) fn flux_y(q: &State4, gamma: f64) -> [f64;4] {
    let primi_state = conserved_to_primitive(&q, gamma);
    let flux1 = q.mom_y;
    let flux2 = q.mom_x*primi_state[2];
    let flux3 = q.mom_y*primi_state[2] + primi_state[3];
    let flux4 = primi_state[2]*(q.e + primi_state[3]);
    [flux1,flux2,flux3,flux4]
}

