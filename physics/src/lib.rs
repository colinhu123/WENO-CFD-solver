use ndarray::{array, Array2};

pub const GAMMA: f64 = 1.4;

#[derive(Debug,Copy,Clone)]
pub enum Direction {
    X,
    Y,
}
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct State {
    pub rho: f64,
    pub mom_x: f64,
    pub mom_y: f64,
    pub e: f64,
}

impl State {
    pub fn new()-> Self {
        Self {
            rho: 0.0,
            mom_x: 0.0,
            mom_y: 0.0,
            e: 0.0,
        }
    }

    pub fn pressure(&self) -> f64 {
        let u = self.mom_x/self.rho;
        let v = self.mom_y/self.rho;
        (GAMMA - 1.0)*(self.e - 0.5*self.rho*(u.powi(2)+v.powi(2)))
    }

    pub fn flux_x(&self)->Self {
        let u = self.mom_x/self.rho;
        let v = self.mom_y/self.rho;
        let p = self.pressure();
        Self {
            rho: self.mom_x,
            mom_x: self.mom_x*u + p,
            mom_y: v*self.mom_x,
            e: u*(self.e + p),
        }
    }

    pub fn flux_y(&self) -> Self {
        let u = self.mom_x/self.rho;
        let v = self.mom_y/self.rho;
        let p = self.pressure();
        Self {
            rho:self.mom_y,
            mom_x:self.mom_y*u,
            mom_y: self.mom_y*v + p,
            e: v*(self.e + p),
        }
    }


    /// Convert conservatives to primitive variables,
    /// Return (u, v, sound speed, enthalpy)

    pub fn con2primi(&self) -> (f64,f64, f64, f64) {
        let p = self.pressure();
        let c = (GAMMA*p/self.rho).sqrt();
        let h = self.e/self.rho + p/self.rho;
        let u = self.mom_x/self.rho;
        let v = self.mom_y/self.rho;
        (u, v, c, h)

    }

    /// Take Roe average of self and state2
    /// Return (u,v,c,h)

    pub fn roe_average(&self, s2: State)-> (f64,f64,f64, f64) {
        let (u1,v1,_c1,h1) = self.con2primi();
        let (u2, v2,_c2,h2) = s2.con2primi();
        let rho1 = self.rho.sqrt();
        let rho2 = s2.rho.sqrt();
        let u = (u1*rho1+u2*rho2)/(rho1+rho2);
        let v = (v1*rho1+v2*rho2)/(rho1+rho2);
        let h = (h1*rho1+h2*rho2)/(rho1+rho2);
        let c=((GAMMA-1.0)*(h-0.5*(u*u+v*v))).sqrt();
        (u,v,c, h)
    }

    /// Build the right-eigenvector matrix `R` of the Euler flux Jacobian,
    /// Roe-averaged between `self` and `s2`, for a given direction.
    ///
    /// Column ordering corresponds to the eigenvalues:
    ///   X-direction: u-c, u, u, u+c
    ///   Y-direction: v-c, v, v, v+c
    pub fn build_r(&self, s2: State, dir: Direction) -> Array2<f64> {
        let (u, v, c, h) = self.roe_average(s2);
        // Roe-averaged density (geometric mean), used by the eigenvector matrix.
        //let rho = (self.rho * s2.rho).sqrt();
        let kin = 0.5 * (u * u + v * v);

        match dir {
            Direction::X => array![
                [1.0, 1.0, 0.0, 1.0],
                [u - c, u, 0.0, u + c],
                [v, v, 1.0, v],
                [h - u * c, kin,  v, h + u * c],
            ],
            Direction::Y => array![
                [1.0, 1.0, 0.0, 1.0],
                [u, u, 1.0, u],
                [v - c, v, 0.0, v + c],
                [h - v * c, kin,  u, h + v * c],
            ],
        }
    }

    /// Build the left-eigenvector matrix `L` of the Euler flux Jacobian,
    /// Roe-averaged between `self` and `s2`, for a given direction.
    ///
    /// Row ordering corresponds to the eigenvalues:
    ///   X-direction: u-c, u, u, u+c
    ///   Y-direction: v-c, v, v, v+c
    pub fn build_l(&self, s2: State, dir: Direction) -> Array2<f64> {
        let (u, v, c, _h) = self.roe_average(s2);
        let g1 = GAMMA - 1.0;
        let kin = 0.5*(u * u + v * v);
        let inv_c2 = 1.0 / (c * c);

        match dir {
            Direction::X => array![
                [0.5 * (g1 * kin * inv_c2 + u / c), -0.5 * (g1 * u * inv_c2 + 1.0 / c), -0.5 * g1 * v * inv_c2, 0.5 * g1 * inv_c2],
                [1.0 - g1 * kin * inv_c2, g1 * u * inv_c2, g1 * v * inv_c2, -g1 * inv_c2],
                [-v, 0.0, 1.0, 0.0],
                [0.5 * (g1 * kin * inv_c2 - u / c), -0.5 * (g1 * u * inv_c2 - 1.0 / c), -0.5 * g1 * v * inv_c2, 0.5 * g1 * inv_c2],
            ],
            Direction::Y => array![
                [0.5 * (g1 * kin * inv_c2 + v / c), -0.5 * g1 * u * inv_c2, -0.5 * (g1 * v * inv_c2 + 1.0 / c), 0.5 * g1 * inv_c2],
                [1.0 - g1 * kin * inv_c2, g1 * u * inv_c2, g1 * v * inv_c2, -g1 * inv_c2],
                [-u, 1.0, 0.0, 0.0],
                [0.5 * (g1 * kin * inv_c2 - v / c), -0.5 * g1 * u * inv_c2, -0.5 * (g1 * v * inv_c2 - 1.0 / c), 0.5 * g1 * inv_c2],
            ],
        }
    }

    pub fn con2char(&self,l:Array2<f64>) -> State {
        let s1 = array![
            [self.rho],
            [self.mom_x],
            [self.mom_y],
            [self.e],
        ];

        let char = l.dot(&s1);

        Self {
            rho: char[[0,0]],
            mom_x: char[[1,0]],
            mom_y: char[[2,0]],
            e: char[[3,0]],
        }
    }

    pub fn char2con(&self, r: Array2<f64>) -> State {
        let s1 = array![
            [self.rho],
            [self.mom_x],
            [self.mom_y],
            [self.e],
        ];

        let char = r.dot(&s1);

        Self {
            rho: char[[0,0]],
            mom_x: char[[1,0]],
            mom_y: char[[2,0]],
            e: char[[3,0]],
        }
    }

}


pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_new_zeroes() {
        let s = State::new();
        assert_eq!(s.rho, 0.0);
        assert_eq!(s.mom_x, 0.0);
        assert_eq!(s.mom_y, 0.0);
        assert_eq!(s.e, 0.0);
    }

    #[test]
    fn test_pressure_zero_velocity() {
        // No motion: pressure should be (gamma-1)*e
        let s = State {
            rho: 2.0,
            mom_x: 0.0,
            mom_y: 0.0,
            e: 5.0,
        };
        let expected = (GAMMA - 1.0) * 5.0; // 0.4 * 5.0 = 2.0
        assert_eq!(s.pressure(), expected);
    }

    #[test]
    fn test_pressure_with_motion() {
        let s = State {
            rho: 2.0,
            mom_x: 1.0,
            mom_y: 3.0,
            e: 5.0,
        };
        // u=0.5, v=1.5, kin = 2.5, p = 0.4*(5-2.5) = 1.0
        assert!((s.pressure() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_flux_x_zero_velocity() {
        // u=v=0 => flux = (0, p, 0, 0)
        let s = State {
            rho: 2.0,
            mom_x: 0.0,
            mom_y: 0.0,
            e: 5.0,
        };
        let p = s.pressure();
        let f = s.flux_x();
        assert_eq!(f.rho, 0.0);
        assert!((f.mom_x - p).abs() < 1e-12);
        assert_eq!(f.mom_y, 0.0);
        assert_eq!(f.e, 0.0);
    }

    #[test]
    fn test_flux_y_zero_velocity() {
        // u=v=0 => flux = (0, 0, p, 0)
        let s = State {
            rho: 2.0,
            mom_x: 0.0,
            mom_y: 0.0,
            e: 5.0,
        };
        let p = s.pressure();
        let f = s.flux_y();
        assert_eq!(f.rho, 0.0);
        assert_eq!(f.mom_x, 0.0);
        assert!((f.mom_y - p).abs() < 1e-12);
        assert_eq!(f.e, 0.0);
    }

    #[test]
    fn test_flux_x_full() {
        let s = State {
            rho: 2.0,
            mom_x: 1.0,
            mom_y: 3.0,
            e: 5.0,
        };
        let f = s.flux_x();
        assert!((f.rho - 1.0).abs() < 1e-12);        // mom_x
        assert!((f.mom_x - 1.5).abs() < 1e-12);      // mom_x*u + p
        assert!((f.mom_y - 1.5).abs() < 1e-12);      // v*mom_x
        assert!((f.e - 3.0).abs() < 1e-12);          // u*(e+p)
    }

    #[test]
    fn test_flux_y_full() {
        let s = State {
            rho: 2.0,
            mom_x: 1.0,
            mom_y: 3.0,
            e: 5.0,
        };
        let f = s.flux_y();
        assert!((f.rho - 3.0).abs() < 1e-12);        // mom_y
        assert!((f.mom_x - 1.5).abs() < 1e-12);      // mom_y*u
        assert!((f.mom_y - 5.5).abs() < 1e-12);      // mom_y*v + p
        assert!((f.e - 9.0).abs() < 1e-12);          // v*(e+p)
    }

    #[test]
    fn test_con2primi() {
        let s = State {
            rho: 2.0,
            mom_x: 1.0,
            mom_y: 3.0,
            e: 5.0,
        };
        let (u, v, c, h) = s.con2primi();
        assert!((u - 0.5).abs() < 1e-12);           // mom_x/rho
        assert!((v - 1.5).abs() < 1e-12);           // mom_y/rho
        // c = sqrt(1.4*p/rho) = sqrt(0.7)
        assert!((c - (GAMMA * 1.0 / 2.0).sqrt()).abs() < 1e-12);
        // h = (e+p)/rho = 6/2 = 3
        assert!((h - 3.0).abs() < 1e-12);
    }


    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
fn test_l_and_r_are_inverses() {
    // Deliberately rho != 1 (rho = 1 hides density inconsistencies in L).
    let s1 = State { rho: 3.0, mom_x: 1.0, mom_y: 2.0, e: 6.0 };
    let s2 = State { rho: 4.0, mom_x: -1.0, mom_y: 1.0, e: 7.0 };

    for dir in [Direction::X, Direction::Y] {
        let l = s1.build_l(s2, dir);
        let r = s1.build_r(s2, dir);

        // Both L·R and R·L must equal the 4x4 identity.
        for (name, prod) in [("L·R", l.dot(&r)), ("R·L", r.dot(&l))] {
            for i in 0..4 {
                for j in 0..4 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (prod[[i, j]] - expected).abs() < 1e-10,
                        "{dir:?} {name} [{i},{j}] = {} (expected {expected})",
                        prod[[i, j]]
                    );
                }
            }
        }
}
}
}
