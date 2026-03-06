use ndarray::{Array3};
use numpy::{PyArray3, PyArray2, PyReadonlyArray3, IntoPyArray};
use pyo3::prelude::*;


mod utils;
mod weno;
mod riemann;

use utils::{con2primi_local, flux_x_local,flux_y_local};



//
// === PyO3 bindings ===
//


#[pyfunction]
fn flux_x_py(py: Python, q: PyReadonlyArray3<f64>, gamma: f64) -> Py<PyArray3<f64>> {
    let qarr = q.as_array();
    let out = utils::flux_x_local(qarr, gamma);
    out.into_pyarray(py).to_owned()
}

#[pyfunction]
fn flux_y_py(py: Python, q: PyReadonlyArray3<f64>, gamma: f64) -> Py<PyArray3<f64>> {
    let qarr = q.as_array();
    let out = utils::flux_y_local(qarr, gamma);
    out.into_pyarray(py).to_owned()
}

#[pyfunction]
fn con2primi_py(py: Python, q: PyReadonlyArray3<f64>, gamma: f64) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let qarr = q.as_array();
    let (p, a, rho, u, v, h) = utils::con2primi_local(qarr, gamma);
    Ok((p.into_pyarray(py).to_owned(), a.into_pyarray(py).to_owned(), rho.into_pyarray(py).to_owned(), u.into_pyarray(py).to_owned(), v.into_pyarray(py).to_owned()))
}

#[pyfunction]
fn weno_x_py(py: Python, u: PyReadonlyArray3<f64>, gamma: f64) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray3<f64>>, Py<PyArray3<f64>>, Py<PyArray3<f64>>)> {
    let uarr = u.as_array();
    let (q_l, f_l, q_r, f_r) = weno::weno_x_reconstruct_local(uarr, gamma);
    Ok((q_l.into_pyarray(py).to_owned(), f_l.into_pyarray(py).to_owned(), q_r.into_pyarray(py).to_owned(), f_r.into_pyarray(py).to_owned()))
}

#[pyfunction]
fn weno_y_py(py: Python, u: PyReadonlyArray3<f64>, gamma: f64) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray3<f64>>, Py<PyArray3<f64>>, Py<PyArray3<f64>>)> {
    let uarr = u.as_array();
    let (q_l, f_l, q_r, f_r) = weno::weno_y_reconstruct_local(uarr, gamma);
    Ok((q_l.into_pyarray(py).to_owned(), f_l.into_pyarray(py).to_owned(), q_r.into_pyarray(py).to_owned(), f_r.into_pyarray(py).to_owned()))
}

#[pyfunction]
fn hllc_x_rs_py(py: Python, 
    q_l: PyReadonlyArray3<f64>, 
    q_r: PyReadonlyArray3<f64>, 
    f_l: PyReadonlyArray3<f64>, 
    f_r: PyReadonlyArray3<f64>, 
    gamma: f64) -> Py<PyArray3<f64>> {
    let out = riemann::hllc_x_local(q_l.as_array(), f_l.as_array(),q_r.as_array(), f_r.as_array(), gamma);
    out.into_pyarray(py).to_owned()
}

#[pyfunction]
fn hllc_y_rs_py(py: Python, 
    q_l: PyReadonlyArray3<f64>, 
    q_r: PyReadonlyArray3<f64>, 
    f_l: PyReadonlyArray3<f64>, 
    f_r: PyReadonlyArray3<f64>, 
    gamma: f64) -> Py<PyArray3<f64>> {
    let out = riemann::hllc_y_local(q_l.as_array(), f_l.as_array(),q_r.as_array(), f_r.as_array(), gamma);
    out.into_pyarray(py).to_owned()
}

#[pyfunction]
fn hlle_x_rs_py(py: Python, 
    q_l: PyReadonlyArray3<f64>, 
    q_r: PyReadonlyArray3<f64>, 
    f_l: PyReadonlyArray3<f64>, 
    f_r: PyReadonlyArray3<f64>, 
    gamma: f64) -> Py<PyArray3<f64>> {
    let out = riemann::hlle_x_local(q_l.as_array(), f_l.as_array(),q_r.as_array(), f_r.as_array(), gamma);
    out.into_pyarray(py).to_owned()
}

#[pyfunction]
fn hlle_y_rs_py(py: Python, 
    q_l: PyReadonlyArray3<f64>, 
    q_r: PyReadonlyArray3<f64>, 
    f_l: PyReadonlyArray3<f64>, 
    f_r: PyReadonlyArray3<f64>, 
    gamma: f64) -> Py<PyArray3<f64>> {
    let out = riemann::hlle_y_local(q_l.as_array(), f_l.as_array(),q_r.as_array(), f_r.as_array(), gamma);
    out.into_pyarray(py).to_owned()
}

#[pyfunction]
fn hllc_hlle_x_rs_py(py: Python, 
    q_l: PyReadonlyArray3<f64>, 
    f_l: PyReadonlyArray3<f64>, 
    q_r: PyReadonlyArray3<f64>, 
    f_r: PyReadonlyArray3<f64>, 
    gamma: f64, 
    jp_lo: f64, 
    jp_hi: f64, 
    force_hlle: bool) -> Py<PyArray3<f64>> {
    let out = if force_hlle {
        riemann::hlle_x_local(q_l.as_array(), f_l.as_array(),q_r.as_array(), f_r.as_array(), gamma)
    } else {
        riemann::hllc_hlle_blend_x_local(q_l.as_array(), f_l.as_array(),q_r.as_array(), f_r.as_array(), gamma, jp_lo, jp_hi)
    };
    out.into_pyarray(py).to_owned()
}

#[pyfunction]
fn hllc_hlle_y_rs_py(py: Python, 
    q_l: PyReadonlyArray3<f64>, 
    f_l: PyReadonlyArray3<f64>, 
    q_r: PyReadonlyArray3<f64>, 
    f_r: PyReadonlyArray3<f64>, 
    gamma: f64, 
    jp_lo: f64, 
    jp_hi: f64, 
    force_hlle: bool) -> Py<PyArray3<f64>> {
    let out = if force_hlle {
        riemann::hlle_y_local(q_l.as_array(), f_l.as_array(),q_r.as_array(), f_r.as_array(), gamma)
    } else {
        riemann::hllc_hlle_blend_y_local(q_l.as_array(), f_l.as_array(),q_r.as_array(), f_r.as_array(), gamma, jp_lo, jp_hi)
    };
    out.into_pyarray(py).to_owned()
}

#[pyfunction]
fn l_local_py<'py>(
    py: Python<'py>,
    u: PyReadonlyArray3<'_, f64>,
    dx: f64,
    gamma: f64,
    force_hlle: bool,
    jp_cri: (f64, f64),
) -> PyResult<&'py PyArray3<f64>> {
    // Get an ndarray view of the input (zero-copy if possible)
    let u_view = u.as_array(); // type: ndarray::ArrayView3<f64>

    // Call the Rust function (keeps the call inside the GIL for safety).
    // If this becomes a performance issue, see notes about py.allow_threads.
    let out: Array3<f64> = riemann::l_local(u_view, dx, gamma, force_hlle, jp_cri);

    // Convert the resulting Array3<f64> into a numpy array to return
    Ok(PyArray3::from_owned_array(py, out))
}


#[pymodule]
fn weno_ext(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(flux_x_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux_y_py, m)?)?;
    m.add_function(wrap_pyfunction!(con2primi_py, m)?)?;
    m.add_function(wrap_pyfunction!(weno_x_py, m)?)?;
    m.add_function(wrap_pyfunction!(weno_y_py, m)?)?;
    m.add_function(wrap_pyfunction!(hllc_x_rs_py, m)?)?;
    m.add_function(wrap_pyfunction!(hllc_y_rs_py, m)?)?;
    m.add_function(wrap_pyfunction!(hlle_x_rs_py, m)?)?;
    m.add_function(wrap_pyfunction!(hlle_y_rs_py, m)?)?;
    m.add_function(wrap_pyfunction!(hllc_hlle_x_rs_py, m)?)?;
    m.add_function(wrap_pyfunction!(hllc_hlle_y_rs_py, m)?)?;
    m.add_function(wrap_pyfunction!(l_local_py, m)?)?;
    Ok(())
}