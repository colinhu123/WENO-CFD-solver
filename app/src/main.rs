use mesh;
use physics;

use mesh::cutcell::{Point,GridInfo};

fn main() {
    println!("Hello, world!");

    let grid = GridInfo {
            nx: 10,
            ny: 10,
            dx: 0.1,
            dy: 0.1,
            x0: 0.0,
            y0: 0.0,
        };
    let p1 = Point {x: 0.05, y: 0.15};
    let p2 = Point {x: 0.35, y: 0.15};
    let p3 = Point {x: 0.35, y: 0.35};
    let p4 = Point {x: 0.05, y: 0.35};

    let poly = mesh::cutcell::Polygon {
        point: vec![p1, p2, p3, p4],
        is_rect: false,
    };

    let (mask, _cellinfo) = mesh::geometry_preprocessing(&poly, &grid);
    let s1 = physics::State { rho: 3.0, mom_x: 1.0, mom_y: 2.0, e: 6.0 };
    let s2 = physics::State { rho: 4.0, mom_x: -1.0, mom_y: 1.0, e: 7.0 };

    for dir in [physics::Direction::X, physics::Direction::Y] {
        let l = s1.build_l(s2, dir);
        let r = s1.build_r(s2, dir);

        let lr = l.dot(&r);
        let rl = r.dot(&l);

        println!("{:?}",lr);
        println!("{:?}",rl);

    }
    
}
