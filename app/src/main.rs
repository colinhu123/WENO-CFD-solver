use mesh;
use physics;
use numerics;
use mesh::cutcell::{Point,GridInfo};

use ndarray::{array, Array2};

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

    let l = s1.build_l(s2,physics::Direction::X);

    let state1 = array![
        [s1.rho],
        [s1.mom_x],
        [s1.mom_y],
        [s1.e],
    ];

    let sten = numerics::weno::WenoStencil {
        points: [s1,s1,s1,s1,s1,s1],
    };

    let res = sten.weno_reconstruction();

    println!("{:?}", res.0);
    println!("{:?}", res.1);

    let char = l.dot(&state1);

    println!("{:?}", s1.con2char(l));

    println!("{:?}", char);

    //println!("{:?}", char[[1,0]]);
    
}
