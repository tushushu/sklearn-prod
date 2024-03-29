# sklearn-prod

[![License](https://img.shields.io/github/license/tushushu/sklearn-prod)](https://github.com/tushushu/sklearn-prod/blob/main/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-flake8-blue)](https://github.com/PyCQA/flake8)  
  

### What
Sklearn-prod is written in Rust([PYO3](https://github.com/PyO3/pyo3)) and provides a Python API. It focuses on model services, and does not provide model training methods. Instead, it reads the parameters of the trained model and generates a new model with smaller size, faster speed and supports multiple input formats.

It provides:
* Support `json` inputs;
* Model which keeps the minimum data required for predictions;
* Highly optimized performance;


### Requirements
* Python: 3.8+    
* OS: Linux, MacOS and Windows
