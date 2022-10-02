use pyo3::exceptions::PyKeyError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct LinearRegression {
    coefs: HashMap<String, f64>,
}

#[pymethods]
impl LinearRegression {
    #[new]
    pub fn new(coefs: HashMap<String, f64>) -> PyResult<Self> {
        if coefs.len() == 0 {
            return Err(PyValueError::new_err("Empty coefs!"));
        }
        Ok(LinearRegression { coefs: coefs })
    }

    pub fn predict(&self, x: HashMap<String, f64>) -> PyResult<f64> {
        if x.len() != self.coefs.len() {
            return Err(PyValueError::new_err(format!(
                "Expected feature size is {}, but got {}!",
                self.coefs.len(),
                x.len(),
            )));
        }
        let mut result: f64 = 0.0;
        for (feat_nm, val) in x.iter() {
            if let Some(coef) = self.coefs.get(feat_nm) {
                result += val * coef;
            } else {
                return Err(PyKeyError::new_err(format!(
                    "Feature name {} is not found!",
                    feat_nm,
                )));
            }
        }
        Ok(result)
    }
}

#[pymodule]
fn sklearn_prod(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LinearRegression>()?;
    Ok(())
}
