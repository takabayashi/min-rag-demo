# Python Object-Oriented Programming FAQ

## What is Object-Oriented Programming (OOP)?

OOP is a programming paradigm that organizes code into objects that contain data and code. It promotes code reusability and maintainability.

## What is a class in Python?

A class is a blueprint for creating objects. It defines the attributes and methods that objects of that class will have.

## How do I define a class in Python?

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

## What is the `__init__` method?

The `__init__` method is a special constructor method that is called when creating a new instance of a class.

## What is `self` in Python?

`self` refers to the instance of the class. It's used to access instance attributes and methods.

## How do I create an object (instance) of a class?

```python
person = Person("John", 30)
```

## What is inheritance in Python?

Inheritance allows a class to inherit attributes and methods from another class (parent class).

## How do I implement inheritance?

```python
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
```

## What is method overriding?

Method overriding occurs when a subclass provides a specific implementation of a method that is already defined in its parent class.

## What are private attributes in Python?

Private attributes are indicated by a double underscore prefix (`__attribute`) and are not directly accessible from outside the class.

## What is encapsulation?

Encapsulation is the bundling of data and methods that operate on that data within a single unit (class).

## What is polymorphism?

Polymorphism allows objects of different classes to be treated as objects of a common superclass.

## What are class methods?

Class methods are methods that are bound to the class rather than an instance. They use the `@classmethod` decorator.

## What are static methods?

Static methods are methods that don't require access to class or instance data. They use the `@staticmethod` decorator.

## What is the difference between instance, class, and static methods?

Instance methods receive `self`, class methods receive `cls`, and static methods receive neither.

## What is the `super()` function?

`super()` is used to call methods from the parent class in inheritance scenarios.

## What are abstract classes?

Abstract classes are classes that cannot be instantiated and are meant to be subclassed.

## How do I create an abstract class in Python?

Use the `abc` module and the `@abstractmethod` decorator.

## What is multiple inheritance?

Multiple inheritance allows a class to inherit from multiple parent classes.

## What is the Method Resolution Order (MRO)?

MRO determines the order in which Python searches for methods in inheritance hierarchies.

## What are properties in Python?

Properties allow you to define getter and setter methods for class attributes using the `@property` decorator.
