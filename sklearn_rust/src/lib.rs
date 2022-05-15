use numpy::ndarray::Array1;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct LinearRegression {
    coef: Array1<f64>,
    col_nm: Vec<String>,
}

#[pymethods]
impl LinearRegression {
    #[new]
    pub fn new(coef: PyReadonlyArray1<f64>, col_nm: Vec<String>) -> Self {
        LinearRegression {
            coef: coef.to_owned_array(),
            col_nm: col_nm,
        }
    }

    pub fn predict_json(&self, x: HashMap<String, f64>) -> f64 {
        x.len();
        self.coef.len();
        self.col_nm.len();
        1.0
    }
}

#[pymodule]
fn sklearn_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LinearRegression>()?;
    Ok(())
}
