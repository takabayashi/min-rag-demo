# Java Basics FAQ

## What is Java?

Java is a high-level, object-oriented programming language developed by Sun Microsystems (now Oracle). It's known for its "Write Once, Run Anywhere" capability.

## What are Java's key features?

Java features include platform independence, object-oriented programming, automatic memory management (garbage collection), and strong type checking.

## How do I install Java?

Download the Java Development Kit (JDK) from Oracle's website or use OpenJDK. Set the JAVA_HOME environment variable.

## What is the difference between JRE and JDK?

JRE (Java Runtime Environment) runs Java applications, while JDK (Java Development Kit) includes JRE plus development tools.

## How do I write a "Hello World" program in Java?

```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

## What are variables in Java?

Variables are containers for storing data values. In Java, you must declare the variable type explicitly.

## What are the primitive data types in Java?

Java has 8 primitive types: byte, short, int, long, float, double, char, and boolean.

## How do I declare a variable in Java?

```java
int number = 10;
String text = "Hello";
double decimal = 3.14;
```

## What is the difference between primitive and reference types?

Primitive types store values directly, while reference types store references to objects.

## What are arrays in Java?

Arrays are fixed-size collections of elements of the same type.

## How do I create an array in Java?

```java
int[] numbers = {1, 2, 3, 4, 5};
String[] names = new String[3];
```

## What are strings in Java?

Strings are objects that represent sequences of characters. They are immutable in Java.

## How do I create a string in Java?

```java
String str1 = "Hello";
String str2 = new String("Hello");
```

## What is the difference between == and .equals() for strings?

== compares object references, while .equals() compares string content.

## What are operators in Java?

Operators are symbols that perform operations on operands. Java has arithmetic, relational, logical, and assignment operators.

## What is the difference between ++i and i++?

++i increments before use (pre-increment), while i++ increments after use (post-increment).

## What are control structures in Java?

Control structures include if-else statements, loops (for, while, do-while), and switch statements.

## How do I write an if-else statement?

```java
if (condition) {
    // code
} else {
    // code
}
```

## What are loops in Java?

Loops allow you to execute a block of code repeatedly. Java supports for, while, and do-while loops.

## How do I use a for loop?

```java
for (int i = 0; i < 10; i++) {
    System.out.println(i);
}
```

## What is the main method?

The main method is the entry point of a Java application. It must be public, static, and void.
