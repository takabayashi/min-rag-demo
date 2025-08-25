# Python Advanced Topics FAQ

## What are decorators in Python?

Decorators are functions that modify the behavior of other functions. They use the `@` syntax and are a form of metaprogramming.

## How do I create a decorator?

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function execution")
        result = func(*args, **kwargs)
        print("After function execution")
        return result
    return wrapper
```

## What are generators in Python?

Generators are functions that return an iterator. They use the `yield` keyword instead of `return`.

## How do I create a generator?

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

## What is the difference between `yield` and `return`?

`yield` pauses function execution and returns a value, while `return` terminates the function.

## What are list comprehensions?

List comprehensions provide a concise way to create lists based on existing sequences.

## How do I use list comprehensions?

```python
squares = [x**2 for x in range(10)]
```

## What are lambda functions?

Lambda functions are small anonymous functions defined with the `lambda` keyword.

## How do I create a lambda function?

```python
add = lambda x, y: x + y
```

## What is the Global Interpreter Lock (GIL)?

The GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python code simultaneously.

## What are context managers?

Context managers are objects that define the methods `__enter__` and `__exit__` for use with the `with` statement.

## How do I create a context manager?

```python
class MyContextManager:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
```

## What are metaclasses?

Metaclasses are classes for classes. They control how classes are created.

## What is monkey patching?

Monkey patching is the practice of modifying classes or modules at runtime.

## What are descriptors?

Descriptors are objects that define `__get__`, `__set__`, or `__delete__` methods.

## What is the difference between `__str__` and `__repr__`?

`__str__` is for readable string representation, while `__repr__` is for detailed representation.

## What are slots in Python?

Slots are a way to optimize memory usage by pre-declaring instance attributes.

## How do I use slots?

```python
class Point:
    __slots__ = ['x', 'y']
```

## What is the difference between `isinstance()` and `type()`?

`isinstance()` checks if an object is an instance of a class or its subclasses, while `type()` returns the exact type.

## What are abstract base classes?

Abstract base classes define a common API for a set of subclasses.

## How do I create an abstract base class?

```python
from abc import ABC, abstractmethod

class MyABC(ABC):
    @abstractmethod
    def my_method(self):
        pass
```

## What is the difference between `__new__` and `__init__`?

`__new__` creates the object, while `__init__` initializes it.
