# Explainable Machine Learning (explainable-ml)
As a student, this repository keeps track of all the ML algorithms I've seen in my lectures and in my research.
This project is my journey to a better understanding of ML algorithms.
Most of the code included is intended to provide a clear explanation; thus it might not be optimized. Hence, some algorithms found here might not be suitable for production.

## Installation
The code was written using Python 3.5 and Numpy. I also use Jupyter Notebook for interactive examples of the algorithms.
0. Not required, by highly suggested, you should create a virtual environment. I use [pyenv](https://github.com/pyenv/pyenv).
1. Install dependencies
```bash
$ pip install -r requirements.txt
```

## Running the example
Because I use this project as a daily reference for my projects and research, all the algorithms have a Jupyter Notebook associated.
1. Run the Jupyter Server
```bash
$ jupyter examples
```
2. In the browser, typically `localhost:8888`, find and launch the notebook.
3. Alternatively, if the example is a python file, run python with a module flag.
```bash
$ python -m src.examples.decision_tree
```

## Running the tests
I try to write UnitTests with every algorithms or utils functions included in this repository.
The UnitTest were writen with the `unittest` framework provided by Python.
1. Run specific test files
```bash
$ python -m unittest src/utils/tests/test_dataset.py ...
```

## Why?
There are many projects (that might even be better than this one!) providing excellent walks through for many ML algorithms, like [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch).
But because I need to implement an algorithm by myself to fully understand it, only reading the code does not help me enough. 
It's also important to mention that I got my inspiration for this repository from existing similar projects.
