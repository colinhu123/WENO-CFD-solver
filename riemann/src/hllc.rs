use physics::{Direction, State};



pub fn hllc(state_l: State, state_r: State, dir: Direction) -> State {

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
            let denom = rho_l*(s_l - u_l) - rho_r*(s_r - u_r);
            let s_star = (p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)) / denom;

            //Left star
            let coeff_l = rho_l * (s_l - u_l) / (s_l - s_star);
            let el_spec = state_l.e / rho_l;
            let e_star_spec_l = el_spec + (s_star - u_l) * (s_star + p_l / (rho_l * (s_l - u_l)));
            let q_star_l = [coeff_l, coeff_l * s_star, coeff_l * v_l, coeff_l * e_star_spec_l];

            // Right star
            let coeff_r = rho_r * (s_r - u_r) / (s_r - s_star);
            let er_spec = state_r.e / rho_r;
            let e_star_spec_r = er_spec + (s_star - u_r) * (s_star + p_r / (rho_r * (s_r - u_r)));
            let q_star_r = [coeff_r, coeff_r * s_star, coeff_r * v_r, coeff_r * e_star_spec_r];

            let mut state_flux = State::new();

            if s_l >= 0.0 {
                state_flux = flux_l;
            } else if s_l < 0.0 && s_star >= 0.0 {
                state_flux.rho = flux_l.rho + s_l * (q_star_l[0] - state_l.rho);
                state_flux.mom_x = flux_l.mom_x + s_l * (q_star_l[1] - state_l.mom_x);
                state_flux.mom_y = flux_l.mom_y + s_l * (q_star_l[2] - state_l.mom_y);
                state_flux.e = flux_l.e + s_l * (q_star_l[3] - state_r.e);            
            } else if s_star < 0.0 && s_r > 0.0 {
                state_flux.rho = flux_r.rho + s_r * (q_star_r[0] - state_r.rho);
                state_flux.mom_x = flux_r.mom_x + s_r * (q_star_r[1] - state_r.mom_x);
                state_flux.mom_y = flux_r.mom_y + s_l * (q_star_r[2] - state_r.mom_y);
                state_flux.e = flux_r.e + s_r * (q_star_l[3] - state_r.e);   
            } else {
                state_flux = flux_r;
            }
            return state_flux
        },
        Direction::Y => {
            let s_l = v - c;
            let s_r = v + c;
            let denom = rho_l * (s_l - v_l) - rho_r * (s_r - v_r);
            let s_star = (p_r - p_l + rho_l * v_l * (s_l - v_l) - rho_r * v_r * (s_r - v_r)) / denom;

            // Left star
            let coeff_l = rho_l * (s_l - v_l) / (s_l - s_star);
            let el_spec = state_l.e / rho_l;
            let e_star_spec_l = el_spec + (s_star - v_l) * (s_star + p_l / (rho_l * (s_l - v_l)));
            let q_star_l = [coeff_l, coeff_l * u_l, coeff_l * s_star, coeff_l * e_star_spec_l];

            // Right star
            let coeff_r = rho_r * (s_r - v_r) / (s_r - s_star);
            let er_spec = state_r.e / rho_r;
            let e_star_spec_r = er_spec + (s_star - v_r) * (s_star + p_r / (rho_r * (s_r - v_r)));
            let q_star_r = [coeff_r, coeff_r * u_r, coeff_r * s_star, coeff_r * e_star_spec_r];
            let mut state_flux = State::new();

            if s_l >= 0.0 {
                state_flux = flux_l;
            } else if s_l < 0.0 && s_star >= 0.0 {
                state_flux.rho = flux_l.rho + s_l * (q_star_l[0] - state_l.rho);
                state_flux.mom_x = flux_l.mom_x + s_l * (q_star_l[1] - state_l.mom_x);
                state_flux.mom_y = flux_l.mom_y + s_l * (q_star_l[2] - state_l.mom_y);
                state_flux.e = flux_l.e + s_l * (q_star_l[3] - state_l.e);            
            } else if s_star < 0.0 && s_r > 0.0 {
                state_flux.rho = flux_r.rho + s_r * (q_star_r[0] - state_r.rho);
                state_flux.mom_x = flux_r.mom_x + s_r * (q_star_r[1] - state_r.mom_x);
                state_flux.mom_y = flux_r.mom_y + s_r * (q_star_r[2] - state_r.mom_y);
                state_flux.e = flux_r.e + s_r * (q_star_r[3] - state_r.e);   
            } else {
                state_flux = flux_r;
            }
            return state_flux
        }
    }


}

