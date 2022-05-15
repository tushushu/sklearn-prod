# sklearn-rust

[![License](https://img.shields.io/github/license/tushushu/ulist)](https://github.com/tushushu/ulist/blob/main/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-flake8-blue)](https://github.com/PyCQA/flake8)  
  

### What
Sklearn-rust is written in Rust (PYO3) and provides a Python API. It focuses on model services, and does not provide model training methods. Instead, it reads the parameters of the trained model and generates a model with a smaller size and a faster prediction speed.


### How
Sklearn-rust provides the following features:
* Multiple `predict` methods support `json`, `list` and `ndarray` inputs;
* Model keeps the minimum data required for predictions;


### Requirements
* Python: 3.8+    
* OS: Linux, MacOS and Windows
