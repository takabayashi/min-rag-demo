# Python Libraries and Frameworks FAQ

## What is NumPy?

NumPy is a fundamental library for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices.

## How do I install NumPy?

```bash
pip install numpy
```

## What is Pandas?

Pandas is a data manipulation and analysis library. It provides data structures like DataFrames and Series.

## How do I create a DataFrame in Pandas?

```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
```

## What is Matplotlib?

Matplotlib is a plotting library for creating static, animated, and interactive visualizations.

## How do I create a simple plot with Matplotlib?

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.show()
```

## What is Django?

Django is a high-level web framework that encourages rapid development and clean, pragmatic design.

## How do I start a Django project?

```bash
django-admin startproject myproject
cd myproject
python manage.py runserver
```

## What is Flask?

Flask is a lightweight web framework for Python. It's more minimalistic than Django.

## How do I create a simple Flask app?

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'
```

## What is TensorFlow?

TensorFlow is an open-source machine learning framework developed by Google.

## What is PyTorch?

PyTorch is an open-source machine learning library developed by Facebook.

## What is Scikit-learn?

Scikit-learn is a machine learning library for Python. It features various classification, regression, and clustering algorithms.

## How do I install Scikit-learn?

```bash
pip install scikit-learn
```

## What is Requests?

Requests is a popular HTTP library for making API calls in Python.

## How do I make a GET request with Requests?

```python
import requests
response = requests.get('https://api.example.com/data')
```

## What is Beautiful Soup?

Beautiful Soup is a library for pulling data out of HTML and XML files.

## What is SQLAlchemy?

SQLAlchemy is a SQL toolkit and Object-Relational Mapping (ORM) library for Python.

## What is Jupyter Notebook?

Jupyter Notebook is an open-source web application that allows you to create and share documents containing live code, equations, visualizations, and narrative text.

## What is Virtualenv?

Virtualenv is a tool to create isolated Python environments.

## How do I create a virtual environment?

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```
