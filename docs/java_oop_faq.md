# Java Object-Oriented Programming FAQ

## What is Object-Oriented Programming (OOP)?

OOP is a programming paradigm that organizes code into objects that contain data and code. Java is built around OOP principles.

## What is a class in Java?

A class is a blueprint for creating objects. It defines the attributes (fields) and methods that objects of that class will have.

## How do I define a class in Java?

```java
public class Person {
    private String name;
    private int age;
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
}
```

## What is a constructor?

A constructor is a special method that is called when creating a new instance of a class. It has the same name as the class.

## How do I create an object (instance) of a class?

```java
Person person = new Person("John", 30);
```

## What is inheritance in Java?

Inheritance allows a class to inherit attributes and methods from another class (parent class) using the `extends` keyword.

## How do I implement inheritance?

```java
public class Student extends Person {
    private String studentId;
    
    public Student(String name, int age, String studentId) {
        super(name, age);
        this.studentId = studentId;
    }
}
```

## What is method overriding?

Method overriding occurs when a subclass provides a specific implementation of a method that is already defined in its parent class.

## What are access modifiers in Java?

Java has four access modifiers: public, private, protected, and default (no modifier).

## What is encapsulation?

Encapsulation is the bundling of data and methods that operate on that data within a single unit (class).

## What is polymorphism?

Polymorphism allows objects of different classes to be treated as objects of a common superclass.

## What is the difference between method overloading and overriding?

Overloading is having multiple methods with the same name but different parameters, while overriding is redefining a method in a subclass.

## What are abstract classes?

Abstract classes are classes that cannot be instantiated and are meant to be subclassed. They can contain abstract methods.

## How do I create an abstract class?

```java
public abstract class Animal {
    public abstract void makeSound();
    
    public void sleep() {
        System.out.println("Sleeping...");
    }
}
```

## What are interfaces in Java?

Interfaces are abstract types that define a contract for classes to implement. They can contain abstract methods and default methods.

## How do I create an interface?

```java
public interface Drawable {
    void draw();
    default void erase() {
        System.out.println("Erasing...");
    }
}
```

## What is the difference between abstract classes and interfaces?

Abstract classes can have constructors and instance variables, while interfaces cannot. A class can implement multiple interfaces but extend only one class.

## What are static methods and variables?

Static members belong to the class rather than instances. They can be accessed without creating an object.

## How do I use static members?

```java
public class MathUtils {
    public static int add(int a, int b) {
        return a + b;
    }
}
// Usage: MathUtils.add(5, 3);
```

## What is the `final` keyword?

The `final` keyword can be applied to classes, methods, and variables to prevent inheritance, overriding, or modification.

## What are packages in Java?

Packages are used to organize classes and avoid naming conflicts. They provide namespace management.

## How do I create a package?

```java
package com.example.myapp;

public class MyClass {
    // class content
}
```
