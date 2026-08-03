use physics::{Direction, State};

pub fn hlle(state_l: State, state_r: State, dir: Direction) -> State {
    let rho_l = state_l.rho;
    let rho_r = state_r.rho;
    let flux_l = state_l.flux_x();
    let flux_r = state_r.flux_x();

    let p_l = state_l.pressure();
    let p_r = state_r.pressure();

    let (u_l, v_l,_c_l,_h_l) = state_l.con2primi();
    let (u_r, v_r, _c_r, _h_r) = state_r.con2primi();

    let (u,v,c,_h) = state_l.roe_average(state_r);

    match dir {
        Direction::X => {
            let s_l = u - c;
            let s_r = u + c;
            let denom = s_r - s_l;

            let mut state_flux = State::new();

            if s_l >= 0.0 {
                state_flux = flux_l;
            }
            else if s_r < 0.0 {
                state_flux = flux_r;
            }
            else {
                state_flux.rho = (s_r*flux_l.rho - s_l * flux_r.rho + s_r*s_l*(state_r.rho - state_l.rho))/denom;
                
            }

            state_flux
        }
        Direction::Y => {

            let s_l = v - c;
            let s_r = v + c;
            let denom = s_r - s_l;

            let mut state_flux = State::new();

            if s_l >= 0.0 {
                state_flux = flux_l;
            }
            else if s_r < 0.0 {
                state_flux = flux_r;
            }
            else {
                state_flux.rho = (s_r*flux_l.rho - s_l * flux_r.rho + s_r*s_l*(state_r.rho - state_l.rho))/denom;
            }

            state_flux
        }
    }


}