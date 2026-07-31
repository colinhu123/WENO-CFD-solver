pub const EPS: f64 = 1e-10;
pub const THRESHOLD: f64 = 0.4;

pub fn close(a: f64, b: f64)-> bool {
    if (a-b).abs() < 1e-10 {
        true
    }else{
        false
    }
}