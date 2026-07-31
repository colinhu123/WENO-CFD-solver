pub mod cutcell;
mod utils;
use ndarray::{Array2};
use std::collections::{HashSet, HashMap};

use crate::cutcell::point_in_convex_polygon;

// Mask value codes for the preprocessing output array.
const SOLID_MARK: usize = 0;     // cell fully inside the body
const FLUID_MARK: usize = 1;     // pure fluid cell, far from the body
const NEAR_CUT_MARK: usize = 2;  // fluid cell whose stencil touches a cut cell

/**
 * This crate is designed to have two output: mask with assigned reconstruction 
 * method and a list of cutted cell
 *  
*/


pub fn is_stencil_touch_cutcell(idx: (usize, usize), cell_chain: &[(usize, usize)]) -> bool {
    // Stencil rectangle centered around cell idx covers
    //   i-2 ..= i+3  and  j-2 ..= j+3  (a 6 x 6 region).
    // Returns true if any cut cell in cell_chain lies inside this rectangle.
    let (i, j) = idx;

    // Adjust the low bound for the central cell; no cell exists below index 0,
    // so saturating subtraction clamps the region to the grid edge.
    let i_lo = i.saturating_sub(2);
    let i_hi = i.saturating_add(3);
    let j_lo = j.saturating_sub(2);
    let j_hi = j.saturating_add(3);

    cell_chain.iter().any(|&(ci, cj)| {
        ci >= i_lo && ci <= i_hi && cj >= j_lo && cj <= j_hi
    })
}



pub fn geometry_preprocessing(poly: &cutcell::Polygon, grid: &cutcell::GridInfo)->(Array2<usize>,Vec<cutcell::CellInfo>){
    /*
     * steo 1: create cut cell vec
     * step 2：generate mask with id and indicator
     */
    let mut poly = poly.clone();
    poly.ensure_ccw();
    let poly = &poly;

    let (cell_chain, point_chain) =cutcell::chain_cut_cell(poly, grid);

    let infos = cutcell::build_cell_info(grid, poly, &cell_chain, &point_chain);

    let nx = grid.nx;
    let ny = grid.ny;

    let mut mask: Array2<usize> = Array2::zeros((nx,ny));

    // Build a HashSet for O(1) membership lookups instead of a linear
    // Vec::contains scan for every grid cell.
    let cell_chain_set: HashSet<(usize, usize)> = cell_chain.iter().copied().collect();

    // Map each cut cell index to the indices of its corresponding CellInfo
    // entries inside `infos`. A single cell may be cut more than once and therefore
    // have multiple CellInfo entries (e.g. cell (1,4) which is cut twice), so each
    // key maps to a Vec of indices.
    let mut cell_to_infos: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (k, info) in infos.iter().enumerate() {
        cell_to_infos.entry(info.idx).or_default().push(k);
    }

    for i in 0..nx{
        for j in 0..ny{
            // 1. Cut cell: cell that the body boundary passes through.
            if cell_chain_set.contains(&(i, j)) {
                // Look up the corresponding CellInfo entries for this cut cell.
                if let Some(info_idxs) = cell_to_infos.get(&(i, j)) {
                    for &k in info_idxs {
                        let info = &infos[k];
                        mask[[i,j]] = info.id;
                    }
                }
                continue;
            }

            // 2. Interior solid cell (its center lies inside the body).
            if point_in_convex_polygon(grid.rect(i, j).center(), poly) {
                mask[[i, j]] = SOLID_MARK;
                continue;
            }

            // 3. Fluid cell whose reconstruction stencil touches a cut cell.
            if is_stencil_touch_cutcell((i, j), &cell_chain) {
                mask[[i, j]] = NEAR_CUT_MARK;
                continue;
            }

            // 4. Pure fluid cell (far from the body boundary).
            mask[[i, j]] = FLUID_MARK;
        }
    }
    (mask, infos)

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stencil_touches_cutcell_inside_region() {
        // Center cell (5,5); cut cell (4,6) lies within (3..=8, 3..=8).
        let cell_chain = vec![(4, 6)];
        assert!(is_stencil_touch_cutcell((5, 5), &cell_chain));
    }

    #[test]
    fn stencil_not_touch_far_cutcell() {
        // Center cell (5,5); cut cell (10,5) is outside i up to 8.
        let cell_chain = vec![(10, 5)];
        assert!(!is_stencil_touch_cutcell((5, 5), &cell_chain));
    }

    #[test]
    fn stencil_boundary_edges() {
        // Cut cell exactly on the corner of the stencil region (i=3, j=8).
        let cell_chain = vec![(3, 8), (8, 3)];
        assert!(is_stencil_touch_cutcell((5, 5), &cell_chain));
    }

    #[test]
    fn stencil_clamps_at_grid_edge() {
        // Center cell (0,0); stencil region clamps to 0..=3.
        let cell_chain = vec![(0, 2)];
        assert!(is_stencil_touch_cutcell((0, 0), &cell_chain));
    }
}

