//! Cut-Cell Grid Geometry Preprocessing Module
//!
//! This module implements the geometric preprocessing of a 2D uniform Cartesian grid
//! that has been intersected by an immersed polygonal body boundary, identifying and
//! processing the resulting "cut cells". This is a common step in Immersed Boundary /
//! Cut-Cell Method CFD mesh generation.
//!
//! Input
//! - `GridInfo`: description of a uniform rectangular grid (origin, cell counts, cell spacing)
//! - `Polygon` (body): the immersed body boundary, given as a sequence of vertices.
//!   `Polygon::ensure_ccw` normalizes the vertex order to counter-clockwise (CCW) if needed.
//!
//! ## Main Pipeline
//! 1. `body.ensure_ccw()`: checks the signed area of the input polygon and reverses the
//!    vertex order if it is not already counter-clockwise, since downstream geometry
//!    routines (normals, convexity, point-in-polygon tests, etc.) assume CCW winding.
//! 2. `chain_cut_cell`: walks along the body boundary edge by edge across the grid,
//!    tracing which grid cells the boundary passes through, and records both the
//!    ordered chain of cell indices and the intersection points where the boundary
//!    crosses each cell's edge. `cut_cell_single` / `edge2rect_edge` / `edge2idx`
//!    determine which side (or corner) of a rectangular cell a ray exits through,
//!    and convert that into the corresponding neighboring cell index.
//! 3. `build_cell_info`: combines the cell chain and intersection points to construct
//!    a `CellInfo` for each cut cell:
//!    - builds the "fluid region" polygon inside the cell (handling two cases —
//!      whether an intersection point coincides with a body vertex or not)
//!    - computes the fluid area and fluid volume fraction (fluid_area / fluid_frac)
//!    - classifies each polygon edge as Fluid (flow-facing) or Wall (solid boundary)
//! 4. `CellInfo::cell_merge`: for each cut cell, checks whether its fluid volume
//!    fraction is below the merge threshold and, if so, picks a candidate neighboring
//!    cell (via its longest fluid edge) to merge into, avoiding numerical instability
//!    caused by very small cut-cell volumes in the solver.
//!
//!  Output
//! - The normalized (CCW) body polygon and its signed area
//! - The ordered sequence of grid cell indices crossed by the body boundary, together
//!   with the intersection point coordinates (returned by `chain_cut_cell`)
//! - A `CellInfo` for each cut cell: cell index, fluid area / fluid fraction, fluid
//!   region polygon, per-edge boundary type, and (after `cell_merge`) whether/where
//!   the cell would be merged
//!
//! `main()` provides an example: an irregular 5-vertex body embedded in a 6x6 grid
//! with spacing 0.8, printing the CCW-normalized body vertices and signed area, the
//! cut-cell chain and intersection points, and the per-cell `CellInfo` both before
//! and after evaluating merge candidates (note: `merged_list` is passed as an empty
//! set here, so each cell's merge decision is evaluated independently rather than
//! propagated/iterated). The reference graph is attached in the project folder 
//! cut_cell_demo.png
//! 
//! It's worth noting the cell (1, 4) which is cutted twice and has two CellInfo in 
//! the list, refering to two independent control volume and needed to be updated 
//! independently


use std::collections::{HashSet, HashMap};
use std::cmp::Ordering;
use crate::utils;

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    fn norm(&self) -> Self {
        let dx = self.x;
        let dy = self.y;
        let length = (dx * dx + dy * dy).sqrt();
        let dx = dx/length;
        let dy = dy/length;
        Self {x: dx, y: dy}
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub start: Point,
    pub end: Point,
}

impl Edge {
    pub fn slope(&self) -> f64 {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        let radians = dy.atan2(dx);
        let degrees = radians.to_degrees();
        let result = degrees.rem_euclid(360.0);
        result
    }

    pub fn dir_vec(&self) -> Point {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        Point {x: dx, y: dy}
    }

    pub fn normal_out(&self) -> Point {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        let l = (dx * dx + dy * dy).sqrt();

        Point {x:dy/l, y: -dx / l}
    }

    pub fn length(&self) -> f64 {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        (dx * dx + dy * dy).sqrt()
    }
}


#[derive(Clone, Debug)]
pub struct Polygon {
    pub point: Vec<Point>,
    pub is_rect: bool
}

impl Polygon {
    pub fn signed_area(&self) -> f64 {
        let poly = &self.point;
        let mut s = 0.0;
        for i in 0..poly.len() {
            let x1 = poly[i].x;
            let y1 = poly[i].y;
            let p1 = poly[(i+1) % poly.len()];
            let x2 = p1.x;
            let y2 = p1.y;

            s += x1 * y2 - x2 * y1;
        }
        0.5 * s
    }

    pub fn ensure_ccw(&mut self) {
        let area = self.signed_area();
        if area < 0.0 {
            self.point.reverse();
        }
    }


    pub fn coord_range(&self) -> (f64,f64,f64,f64) {
        if self.is_rect {
            (self.point[0].x,self.point[0].y,self.point[2].x,self.point[2].y)
        }
        else {
            (0.0,0.0,0.0,0.0)
        }
    }

    pub fn len(&self) -> usize {
        self.point.len()
    }

    pub fn edge_list(&self) -> Vec<Edge> {
        let mut target = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            target.push(Edge {
                start: self.point[i],
                end: self.point[(i+1) % self.len()]
            })
        }
        target
    }

    pub fn is_convex(&self) -> bool {
        let n = self.point.len();

        if n < 3 {
            return false;
        }

        let mut sign = 0i32;

        for i in 0..n {
            let a = self.point[i];
            let b = self.point[(i + 1) % n];
            let c = self.point[(i + 2) % n];

            let cross =
                (b.x - a.x) * (c.y - b.y)
              - (b.y - a.y) * (c.x - b.x);

            // 忽略共线点
            if cross.abs() < 1e-12 {
                continue;
            }

            let current_sign = if cross > 0.0 { 1 } else { -1 };

            if sign == 0 {
                sign = current_sign;
            } else if sign != current_sign {
                return false;
            }
        }

        true
    }

    pub fn center(&self) -> Point {
        let n = self.point.len();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for i in 0..n {
            sum_x += self.point[i].x;
            sum_y += self.point[i].y;
        }
        let n = n as f64;
        Point {x: sum_x/n, y: sum_y/n}
    }
}

pub struct GridInfo {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub x0: f64,
    pub y0: f64,
}

impl GridInfo {
    pub fn rect(&self, i: usize, j: usize)-> Polygon {
        let i = i as f64;
        let j = j as f64;
        let p1 = Point {x: self.x0 + i*self.dx, y: self.y0+j*self.dy};
        let p2 = Point {
            x: self.x0 + (i+1.0)*self.dx,
            y: self.y0 + j*self.dy,
        };
        let p3 = Point {
            x: self.x0 + (i+1.0)*self.dx,
            y: self.y0 + (j+1.0)*self.dy,
        };
        let p4 = Point {
            x: self.x0 + i*self.dx,
            y: self.y0 + (j+1.0)*self.dy,
        };
        let poly = vec![p1,p2,p3,p4];
        let poly = Polygon {point: poly, is_rect: true};
        poly
    }

    pub fn locate_point(&self, p:Point) -> (usize,usize){
        let i = ((p.x-self.x0)/self.dx) as usize;
        let j = ((p.y-self.y0)/self.dy) as usize;
        (i,j)
    }
}
#[derive(Debug,PartialEq)]
pub enum RectEdge {
    Left,
    Right,
    Bottom,
    Top,
    TLCorner,
    TRCorner,
    BLCorner,
    BRCorner,
}

pub fn edge2idx(rect_edge: RectEdge, idx: (usize, usize)) -> (usize,usize) {
    match rect_edge {
        RectEdge::Right => (idx.0+1,idx.1),
        RectEdge::Left  => (idx.0 - 1, idx.1),
        RectEdge::Top   => (idx.0, idx.1 + 1),
        RectEdge::Bottom => (idx.0, idx.1 - 1),
        RectEdge::TRCorner => (idx.0 + 1, idx.1 + 1),
        RectEdge::BRCorner => (idx.0 + 1, idx.1 - 1),
        RectEdge::TLCorner => (idx.0 - 1, idx.1 + 1),
        RectEdge::BLCorner => (idx.0 - 1, idx.1 - 1),
    }
}

pub fn edge2rect_edge(edge: Edge, rect: &Polygon) -> RectEdge {
    let angle = edge.slope();
    let (xmin,ymin,_xmax,_ymax) = rect.coord_range();

    if utils::close(angle,90.0) || utils::close(angle, 270.0) {
        if edge.start.x == xmin {
            RectEdge::Left
        }
        else {
            RectEdge::Right
        }
    }
    else {
        if edge.start.y == ymin {
            RectEdge::Bottom
        }
        else {
            RectEdge::Top
        }
    }
}


pub fn point_in_convex_polygon(p: Point, poly: &Polygon) -> bool {
    /*
    True -> in polygon;
    False -> outof polygon
    */
    let mut worst = f64::NEG_INFINITY;

    for edge in poly.edge_list() {
        let n = edge.normal_out();
        let val = (p.x - edge.start.x) * n.x + (p.y - edge.start.y) * n.y;
        worst = worst.max(val);
    }

    worst <= utils::EPS
}

pub fn cut_cell_single(rect: &Polygon,e: Edge,p: Point) -> (Point, RectEdge){
    let dir = e.dir_vec();
    let dir = dir.norm();
    let angle = e.slope();
    let (xmin,ymin,xmax,ymax) = rect.coord_range();
    if utils::close(angle, 0.0) || utils::close(angle, 360.0) {
        return (Point {x: xmax, y: p.y}, RectEdge::Right)
    }
    if utils::close(angle,90.0) {
        return (Point {x:p.x,y: ymax},RectEdge::Top)
    }
    if utils::close(angle,180.0) {
        return (Point {x: xmin, y: p.y}, RectEdge::Left)
    }
    if utils::close(angle, 270.0) {
        return (Point {x: p.x, y: ymin}, RectEdge::Bottom)
    }
    if angle > 0.0 && angle < 90.0 {
        let alpha = (ymax-p.y)/dir.y;
        let beta = (xmax-p.x)/dir.x;
        if utils::close(alpha, beta) {
            return (Point {x: p.x + beta*dir.x, y: p.y + beta * dir.y},RectEdge::TRCorner)
        }
        if alpha > beta {
            return (Point {x: p.x + beta*dir.x, y: p.y + beta * dir.y}, RectEdge::Right)
            
        }
        
        else {
            return (Point {x: p.x + alpha*dir.x, y: p.y + alpha * dir.y}, RectEdge::Top)
        }
    }
    if angle > 90.0 && angle < 180.0 {
        let alpha = (ymax-p.y)/dir.y;
        let beta = (xmin-p.x)/dir.x;
        println!("{}",alpha);
        println!("{}",beta);
        if utils::close(alpha, beta) {
            return (Point {x: p.x + beta*dir.x, y: p.y + beta * dir.y},RectEdge::TLCorner)
        }
        if alpha > beta {
            return (Point {x: p.x + beta*dir.x, y: p.y + beta * dir.y}, RectEdge::Left)
            
        }
        else {
            return (Point {x: p.x + alpha*dir.x, y: p.y + alpha * dir.y}, RectEdge::Top)
        }
    }
    if angle > 180.0 && angle < 270.0{
        let alpha = (ymin-p.y)/dir.y;
        let beta = (xmin-p.x)/dir.x;
        if utils::close(alpha, beta) {
            return (Point {x: p.x + beta*dir.x, y: p.y + beta * dir.y}, RectEdge::BLCorner)
        }
        if alpha > beta {
            return (Point {x: p.x + beta*dir.x, y: p.y + beta * dir.y}, RectEdge::Left)
            
        }
        else {
            return (Point {x: p.x + alpha*dir.x, y: p.y + alpha * dir.y}, RectEdge::Bottom)
        }
    }
    else {//Case for which angle is from 270 to 360
        let alpha = (ymin-p.y)/dir.y;
        let beta = (xmax-p.x)/dir.x;

        if utils::close(alpha, beta) {
            return (Point {x: p.x + beta*dir.x, y: p.y + beta * dir.y}, RectEdge::BRCorner)
        }
        if alpha > beta {
            return (Point {x: p.x + beta*dir.x, y: p.y + beta * dir.y}, RectEdge::Right)
            
        }
        else {
            return (Point {x: p.x + alpha*dir.x, y: p.y + alpha * dir.y}, RectEdge::Bottom)
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundType {
    Fluid,
    Wall,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellType {
    Cut,
    Fluid,
    Solid,
}

pub fn boundary_s(rect: &Polygon, p: Point) -> f64 {
    let (xmin, ymin, xmax, ymax) = rect.coord_range();
    let w = xmax - xmin;
    let h = ymax - ymin;
    let eps = 1e-12;

    // 四个角先特判，避免歧义
    if (p.x - xmin).abs() < eps && (p.y - ymin).abs() < eps {
        return 0.0; // BL
    }
    if (p.x - xmax).abs() < eps && (p.y - ymin).abs() < eps {
        return w; // BR
    }
    if (p.x - xmax).abs() < eps && (p.y - ymax).abs() < eps {
        return w + h; // TR
    }
    if (p.x - xmin).abs() < eps && (p.y - ymax).abs() < eps {
        return 2.0 * w + h; // TL
    }

    // 边上点
    if (p.y - ymin).abs() < eps {
        return p.x - xmin; // bottom: BL -> BR
    }
    if (p.x - xmax).abs() < eps {
        return w + (p.y - ymin); // right: BR -> TR
    }
    if (p.y - ymax).abs() < eps {
        return w + h + (xmax - p.x); // top: TR -> TL
    }

    // left: TL -> BL
    2.0 * w + h + (ymax - p.y)
}

pub fn boundary_points_between_ccw(rect: &Polygon, a: Point, b: Point) -> Vec<Point> {
    let (xmin, ymin, xmax, ymax) = rect.coord_range();
    let w = xmax - xmin;
    let h = ymax - ymin;
    let eps = 1e-12;

    let sa = boundary_s(rect, a);
    let sb = boundary_s(rect, b);

    let corners = [
        (0.0, Point { x: xmin, y: ymin }),         // BL
        (w, Point { x: xmax, y: ymin }),           // BR
        (w + h, Point { x: xmax, y: ymax }),       // TR
        (2.0 * w + h, Point { x: xmin, y: ymax }), // TL
    ];

    let mut pts = Vec::new();

    if sa <= sb {
        for &(s, c) in &corners {
            if s > sa + eps && s < sb - eps {
                pts.push(c);
            }   
        }
    } else {
    // 先收集 s > sa 的部分（按 s 升序，本身就是 CCW 顺序）
        for &(s, c) in &corners {
            if s > sa + eps {
                pts.push(c);
            }
        }
    // 再收集 s < sb 的部分（按 s 升序）
        for &(s, c) in &corners {
            if s < sb - eps {
                pts.push(c);
            }
        }
    }

    pts
}
#[derive(Debug)]
pub struct CellInfo {
    pub idx: (usize, usize),
    pub id: usize,
    pub fluid_area: f64,
    pub fluid_frac: f64,
    pub poly: Polygon,
    pub bound_type: Vec<BoundType>,
    pub cell_type: CellType,
    pub is_merged: bool,
    pub master: (usize,usize),
}

impl CellInfo {
    pub fn vertex_poly(rect: &Polygon, pin: Point, pout: Point, pver: Point, body: &Polygon)-> Polygon {
        let mut poly1: Vec<Point> = Vec::new();
        poly1.push(pin);
        poly1.push(pver);
        poly1.push(pout);

        // 从 pout 沿矩形边界逆时针回到 pin 的中间角点
        let back_path = boundary_points_between_ccw(rect, pout, pin);
        poly1.extend(back_path);

        // 另一块：pin -> ... -> pout -> pver
        let mut poly2: Vec<Point> = Vec::new();
        poly2.push(pin);

        let front_path = boundary_points_between_ccw(rect, pin, pout);
        poly2.extend(front_path);
        poly2.push(pout);
        poly2.push(pver);

        let poly1 = Polygon {point: poly1, is_rect: false};
        let poly2 = Polygon {point: poly2, is_rect: false};

        if poly1.is_convex() {
            if point_in_convex_polygon(poly1.center(),&body) {
                poly2
            }
            else {
                poly1
            }
        }
        else {
            if point_in_convex_polygon(poly2.center(),&body) {
                poly1
            }
            else {
                poly2
            }
        }
        
    }

    pub fn build_vertex_info(rect: &Polygon,
        pin: Point,
        pout: Point, 
        pver: Point, 
        body: &Polygon, idx: (usize,usize),id: usize)-> Self {
            let poly = CellInfo::vertex_poly(rect, pin, pout, pver, body);
            let n = poly.len();
            let mut bound_type: Vec<BoundType> = vec![BoundType::Fluid; n];
            bound_type[n-1] = BoundType::Wall;
            bound_type[n-2] = BoundType::Wall;

            let fluid_area = poly.signed_area().abs();

            Self {
                idx: idx,
                id: id,
                fluid_area: fluid_area,
                fluid_frac: fluid_area/rect.signed_area().abs(),
                poly: poly,
                bound_type: bound_type,
                cell_type: CellType::Cut,
                is_merged: false,
                master: (0,0),
            }
        }

    pub fn nor_poly(rect: &Polygon, pin: Point, pout: Point, body: &Polygon) -> Polygon {
        let mut poly1: Vec<Point> = Vec::new();
        poly1.push(pin);
        let path1 = boundary_points_between_ccw(rect, pin, pout);
        poly1.extend(path1);
        poly1.push(pout);

        // 候选 2：pin -> pout -> ... boundary ... -> pin
        let mut poly2: Vec<Point> = Vec::new();
        poly2.push(pin);
        poly2.push(pout);
        let path2 = boundary_points_between_ccw(rect, pout, pin);
        poly2.extend(path2);

        let poly1 = Polygon { point: poly1, is_rect: false };
        let poly2 = Polygon { point: poly2, is_rect: false };
        if point_in_convex_polygon(poly1.center(), &body) {
            poly2
        }
        else {
            poly1
        }
    }

    pub fn build_nor_info(rect: &Polygon, 
        pin: Point, 
        pout: Point, 
        body: &Polygon, 
        idx: (usize,usize), id: usize)-> Self {
            let poly = CellInfo::nor_poly(rect, pin, pout, body);
            let n = poly.len();
            let mut bound_type: Vec<BoundType> = vec![BoundType::Fluid; n];
            bound_type[n-1] = BoundType::Wall;

            let fluid_area = poly.signed_area().abs();
            Self {
                idx: idx,
                id: id,
                fluid_area: fluid_area,
                fluid_frac: fluid_area/rect.signed_area().abs(),
                poly: poly,
                bound_type: bound_type,
                cell_type: CellType::Cut,
                is_merged: false,
                master:(0,0),
            }
    }

    pub fn fluid_edges_sorted(&self, grid: &GridInfo) -> Vec<(usize, Edge, RectEdge, f64)> {
        let mut edges = Vec::new();
        let edge_list = self.poly.edge_list();
        for i in 0..self.poly.len() {
            if self.bound_type[i] != BoundType::Fluid {
                continue;
            }

            let e = edge_list[i];

            let rect_edge = edge2rect_edge(e, &grid.rect(self.idx.0, self.idx.1));

            edges.push((i,e,rect_edge,e.length()));
        }

        edges.sort_by(|a, b| {
            b.3.partial_cmp(&a.3).unwrap_or(Ordering::Equal)
        });

        edges
    }

    pub fn cell_merge(&self,grid: &GridInfo, merged_list: HashSet<(usize,usize)>) -> Self {
        let mut out = CellInfo {
            idx: self.idx,
            id: self.id,
            fluid_area: self.fluid_area,
            fluid_frac: self.fluid_frac,
            poly: self.poly.clone(),
            bound_type: self.bound_type.clone(),
            cell_type: self.cell_type,
            is_merged: false,
            master: (0, 0),
        };

        out.is_merged = false;
        out.master = (0, 0);

        if self.cell_type != CellType::Cut {
            return out;
        }

        if self.fluid_frac > utils::THRESHOLD {
            return out;
        }

        let fluid_edges = self.fluid_edges_sorted(grid);

        for (_, _edge, rect_edge, _) in fluid_edges {
            let candidate = edge2idx(rect_edge, self.idx);

            if candidate == self.idx {
                continue;
            }

            if merged_list.contains(&candidate) {
                continue;
            }

            out.is_merged = true;
            out.master = candidate;
            return out;
        }

        out
    }
}



fn push_if_new(chain: &mut Vec<(usize, usize)>, idx: (usize, usize)) {
    if chain.last().copied() != Some(idx) {
        chain.push(idx);
    }
}

pub fn chain_cut_cell(body: &Polygon, grid: &GridInfo) -> (Vec<(usize, usize)>, Vec<Point>) {
    let n = body.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut chain: Vec<(usize, usize)> = Vec::new();
    let mut intersect_point: Vec<Point> = Vec::new();

    for k in 0..n {
        let p = body.point[k];
        let q = body.point[(k + 1) % n];
        let edge = Edge { start: p, end: q };

        let target_idx = grid.locate_point(q);
        let mut current_point = p;
        let mut current_idx = grid.locate_point(current_point);

        let mut visited: HashSet<(usize, usize)> = HashSet::new();

        loop {
            // Stop if we've already processed this cell for this edge.
            if !visited.insert(current_idx) {
                break;
            }

            push_if_new(&mut chain, current_idx);

            // If the edge endpoint is already in this cell, we're done.
            if current_idx == target_idx {
                break;
            }

            let rect = grid.rect(current_idx.0, current_idx.1);
            let (exit_point, rect_edge) = cut_cell_single(&rect, edge, current_point);
            let next_idx = edge2idx(rect_edge, current_idx);

            // Guard against no-progress cases.
            if next_idx == current_idx {
                break;
            }

            intersect_point.push(exit_point);

            push_if_new(&mut chain, next_idx);

            if next_idx == target_idx {
                break;
            }

            current_idx = next_idx;
            current_point = exit_point;
        }
    }
    chain.pop();

    (chain, intersect_point)
}



pub fn build_cell_info(
    grid: &GridInfo,
    body: &Polygon,
    cell_chain: &Vec<(usize, usize)>,
    point_chain: &Vec<Point>,
)-> Vec<CellInfo> {
    if cell_chain.len() != point_chain.len() || cell_chain.is_empty() {
        return Vec::new();
    }

    let n = cell_chain.len();

    let mut id: usize = 10;

    let mut cell_to_vertex: HashMap<(usize, usize), Point> = HashMap::new();
    for &v in &body.point {
        let idx = grid.locate_point(v);
        cell_to_vertex.entry(idx).or_insert(v);
    }

    let mut infos: Vec<CellInfo> = Vec::with_capacity(n);

    for i in 0..n {
        let idx = cell_chain[i];
        let p_out = point_chain[i];
        let p_in = point_chain[(i + n - 1) % n];

        if let Some(&p_ver) = cell_to_vertex.get(&idx) {
            infos.push(CellInfo::build_vertex_info(&grid.rect(idx.0, idx.1), p_in, p_out,p_ver,&body,idx, id));
            id += 1;
        } else {
            infos.push(CellInfo::build_nor_info(&grid.rect(idx.0, idx.1), p_in, p_out,&body,idx, id));
            id += 1;
        }
    }

    infos
}

/* 
fn main() {
    // 对应 Python:
    // grid = build_grid(0.0, 0.0, nx=6, ny=6, dx=0.8, dy=0.8)
    let grid = GridInfo {
        _nx: 6,
        _ny: 6,
        dx: 0.8,
        dy: 0.8,
        x0: 0.0,
        y0: 0.0,
    };

    // 对应 Python:
    // body = ensure_ccw([...])
    let mut body = Polygon {
        point: vec![
            Point { x: 1.25, y: 1.15 },
            Point { x: 2.00, y: 0.20 },
            Point { x: 2.70, y: 1.30 },
            Point { x: 2.45, y: 2.50 },
            Point { x: 1.10, y: 4.20 },
        ],
        is_rect: false,
    };

    body.ensure_ccw();

    println!("body signed area = {}", body.signed_area());
    println!("body points after ensure_ccw:");
    for p in &body.point {
        println!("{:?}", p);
    }

    let (cell_chain, point_chain) = chain_cut_cell(&body, &grid);

    println!("cell_chain length = {}", cell_chain.len());
    println!("point_chain length = {}", point_chain.len());

    println!("cell_chain:");
    for idx in &cell_chain {
        println!("{:?}", idx);
    }

    println!("intersection points:");
    for p in &point_chain {
        println!("{:?}", p);
    }

    let infos = build_cell_info(&grid, &body, &cell_chain, &point_chain);

    
    let merged_list: HashSet<(usize, usize)> = HashSet::new();
    let merged_infos: Vec<CellInfo> = infos
        .iter()
        .map(|c| c.cell_merge(&grid,merged_list.clone()))
        .collect();

    for info in &merged_infos {
        println!("{:?}", info);
    }
}

*/



#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < TOL, "left={}, right={}", a, b);
    }

    #[test]
    fn point_norm_should_have_unit_length() {
        let p = Point { x: 3.0, y: 4.0 };
        let q = p.norm();
        let len = (q.x * q.x + q.y * q.y).sqrt();
        assert_close(len, 1.0);
        assert_close(q.x, 0.6);
        assert_close(q.y, 0.8);
    }

    #[test]
    fn edge_slope_basic_directions() {
        let e = Edge {
            start: Point { x: 0.0, y: 0.0 },
            end: Point { x: 1.0, y: 0.0 },
        };
        assert_close(e.slope(), 0.0);

        let e = Edge {
            start: Point { x: 0.0, y: 0.0 },
            end: Point { x: 0.0, y: 1.0 },
        };
        assert_close(e.slope(), 90.0);

        let e = Edge {
            start: Point { x: 0.0, y: 0.0 },
            end: Point { x: -1.0, y: 0.0 },
        };
        assert_close(e.slope(), 180.0);

        let e = Edge {
            start: Point { x: 0.0, y: 0.0 },
            end: Point { x: 0.0, y: -1.0 },
        };
        assert_close(e.slope(), 270.0);
    }

    #[test]
    fn polygon_signed_area_and_ccw() {
        let mut poly = Polygon {
            point: vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: 1.0, y: 0.0 },
                Point { x: 1.0, y: 1.0 },
                Point { x: 0.0, y: 1.0 },
            ],
            is_rect: false,
        };
        assert_close(poly.signed_area(), 1.0);

        poly.point.reverse();
        assert!(poly.signed_area() < 0.0);

    }

    #[test]
    fn grid_rect_should_generate_correct_corners() {
        let grid = GridInfo {
            nx: 10,
            ny: 10,
            dx: 0.1,
            dy: 0.1,
            x0: 0.0,
            y0: 0.0,
        };

        let rect = grid.rect(3, 4);
        assert_close(rect.point[0].x, 0.3);
        assert_close(rect.point[0].y, 0.4);
        assert_close(rect.point[2].x, 0.4);
        assert_close(rect.point[2].y, 0.5);
        assert!(rect.is_rect);
    }

    #[test]
    fn locate_point_should_find_cell() {
        let grid = GridInfo {
            nx: 10,
            ny: 10,
            dx: 0.1,
            dy: 0.1,
            x0: 0.0,
            y0: 0.0,
        };

        let (i, j) = grid.locate_point(Point { x: 0.34, y: 0.47 });
        assert_eq!(i, 3);
        assert_eq!(j, 4);
    }
    #[test]
    fn cut_cell_boundary(){
        let grid = GridInfo {
            nx: 10,
            ny: 10,
            dx: 0.1,
            dy: 0.1,
            x0: 0.0,
            y0: 0.0,
        };

        let rect = grid.rect(3, 3);

        let p1 = Point {x: 0.35, y: 0.35};
        let p2 = Point {x: 0.7, y: 0.7};
        let p3 = Point {x:0.35, y: 0.7};
        let p4 = Point {x: 0.0, y: 0.7};

        let e1 = Edge {start:p1, end: p2};
        let e2 = Edge {start: p1, end: p3};
        let e3 = Edge {start: p1, end: p4};

        let (point,rect_edge) = cut_cell_single(&rect, e1, p1);
        assert_eq!(point.x,0.4);
        assert_eq!(rect_edge,RectEdge::TRCorner);
        assert_eq!(point.y, 0.4);
        let (point, rect_edge) = cut_cell_single(&rect, e2, p1);
        assert_eq!(point.x, 0.35);
        assert_eq!(rect_edge, RectEdge::Top);
        let (_point,rect_edge) = cut_cell_single(&rect, e3, p1);
        assert_eq!(rect_edge, RectEdge::TLCorner);

    }

    #[test]
    fn chain_cut_cell_test_case() {
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

        let poly = Polygon {point: vec![p1,p2,p3,p4], is_rect: true};

        let vec1 = chain_cut_cell(&poly, &grid);
        assert!(vec1.0.contains(&(1,1)));
    }

    #[test]
    fn vertex_poly_test() {
        let grid = GridInfo {
            nx: 10,
            ny: 10,
            dx: 0.1,
            dy: 0.1,
            x0: 0.0,
            y0: 0.0,
        };

        // cell (3,1)
        let rect = grid.rect(3, 1);
        let p1 = Point {x: 0.05, y: 0.15};
            let p2 = Point {x: 0.35, y: 0.15};
            let p3 = Point {x: 0.35, y: 0.35};
            let p4 = Point {x: 0.05, y: 0.35};

            let body = Polygon {point: vec![p1,p2,p3,p4], is_rect: true};

        // 左边进入
        let pin = Point {
            x: 0.3,
            y: 0.15,
        };

        // 上边出去
        let pout = Point {
            x: 0.35,
            y: 0.2,
        };

        // body 顶点
        let pver = Point {
            x: 0.35,
            y: 0.15,
        };

        let poly = CellInfo::vertex_poly(&rect, pin, pout, pver,&body);

        // 三个关键点必须都在 polygon 中
        assert!(poly.point.iter().any(|p| {
            (p.x - pin.x).abs() < 1e-12 &&
            (p.y - pin.y).abs() < 1e-12
        }));

        assert!(poly.point.iter().any(|p| {
            (p.x - pout.x).abs() < 1e-12 &&
            (p.y - pout.y).abs() < 1e-12
        }));

        assert!(poly.point.iter().any(|p| {
            (p.x - pver.x).abs() < 1e-12 &&
            (p.y - pver.y).abs() < 1e-12
        }));
        for point in &poly.point {
            println!("{:?}", point);
        }
        // 应该是凸多边形
        assert!(!poly.is_convex());

        // 面积应大于0
        assert!(poly.signed_area() > 0.0);

        // 中心点应该在 polygon 内
        assert!(point_in_convex_polygon(poly.center(), &body));

        println!("{:?}", poly.point);
    }

}