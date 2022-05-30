use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct LinearRegression {
    mapping: HashMap<String, f64>,
}

#[pymethods]
impl LinearRegression {
    #[new]
    pub fn new(coef: PyReadonlyArray1<f64>, col_nm: Vec<&str>) -> Self {
        let mapping: HashMap<String, f64> = col_nm
            .iter()
            .zip(coef.to_owned_array().iter())
            .map(|(key, val)| (key.to_string(), *val))
            .collect();
        LinearRegression { mapping: mapping }
    }

    pub fn predict_json(&self, x: HashMap<String, f64>) -> f64 {
        if x.len() != self.mapping.len() {
            panic!(format!("Expect the length of x is {}!", self.mapping.len()));
        }
        let mut ret: f64 = 0.0;
        for (key, val) in x.iter() {
            ret += val * self.mapping.get(key).unwrap();
        }
        ret
    }
}

#[pymodule]
fn sklearn_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LinearRegression>()?;
    Ok(())
}
