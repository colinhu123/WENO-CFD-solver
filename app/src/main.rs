use mesh;

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

    println!{"{:?}", mask};
}
