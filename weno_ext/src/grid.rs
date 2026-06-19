use ndarray::{Array3};
use crate::state;

pub(crate) struct Grid {
    nx: usize,
    ny: usize,
    ng: usize,
    dx: f64,
    dy: f64
}

impl Grid {
    pub fn new(nx: usize, ny: usize, ng: usize,dx: f64, dy: f64) -> Self{
        Grid {
            nx: nx,
            ny: ny,
            ng: ng,
            dx: dx,
            dy: dy
        }
    }

    pub fn shape_with_ghoasts(&self,nvar: usize) -> (usize, usize, usize) {
        (self.nx+2*self.ng,self.ny+2*self.ng, nvar)
    }

    pub fn interior_slice(&self) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
        (self.ng..self.ng + self.nx, self.ng..self.ng + self.ny)
    }
}


pub struct Field {
    pub grid: Grid,
    pub q: Array3<f64>,
}