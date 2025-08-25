# Python Best Practices and Debugging FAQ

## What is PEP 8?

PEP 8 is the style guide for Python code. It provides conventions for writing readable code.

## What are the main PEP 8 guidelines?

Use 4 spaces for indentation, limit lines to 79 characters, use descriptive variable names, and follow naming conventions.

## How do I write a docstring?

Docstrings are string literals that appear as the first statement in a module, function, class, or method definition.

## What is the difference between a comment and a docstring?

Comments are for developers, while docstrings are for users of your code and can be accessed programmatically.

## How do I handle exceptions properly?

Use specific exception types, avoid bare except clauses, and provide meaningful error messages.

## What is the proper way to use try-except?

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
```

## How do I use logging in Python?

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.info("This is an info message")
```

## What is the difference between print and logging?

Logging provides more control over output, different levels, and can be configured for different outputs.

## How do I debug Python code?

Use the `pdb` module, print statements, or an IDE debugger.

## How do I use the Python debugger?

```python
import pdb
pdb.set_trace()  # This will pause execution
```

## What are unit tests?

Unit tests are automated tests that verify individual units of code work correctly.

## How do I write unit tests in Python?

Use the `unittest` module or `pytest` framework.

## What is the difference between `unittest` and `pytest`?

`pytest` is more modern and has simpler syntax, while `unittest` is part of the standard library.

## How do I use virtual environments?

Virtual environments isolate project dependencies to avoid conflicts.

## What is the difference between `pip` and `conda`?

`pip` is for Python packages, while `conda` can manage packages for multiple languages.

## How do I manage dependencies?

Use `requirements.txt` files to specify package versions.

## What is the purpose of `__init__.py` files?

`__init__.py` files mark directories as Python packages.

## How do I structure a Python project?

Organize code into modules and packages, separate concerns, and follow the principle of separation of concerns.

## What is the difference between a module and a package?

A module is a single file, while a package is a directory containing multiple modules.

## How do I handle configuration in Python?

Use environment variables, configuration files, or dedicated configuration management libraries.

## What are some common Python anti-patterns?

Avoid mutable default arguments, don't use bare except clauses, and don't ignore exceptions.
