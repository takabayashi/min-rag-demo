# Java Exceptions and Multithreading FAQ

## What are exceptions in Java?

Exceptions are events that occur during the execution of a program that disrupt the normal flow of instructions.

## What is the difference between checked and unchecked exceptions?

Checked exceptions must be handled at compile time, while unchecked exceptions (RuntimeException and its subclasses) do not need to be explicitly handled.

## What are the main exception types in Java?

The main types are Exception (checked), RuntimeException (unchecked), and Error (unchecked, serious problems).

## How do I handle exceptions in Java?

Use try-catch blocks to handle exceptions gracefully.

## How do I write a try-catch block?

```java
try {
    // code that might throw an exception
} catch (ExceptionType e) {
    // handle the exception
} finally {
    // code that always executes
}
```

## What is the finally block?

The finally block contains code that always executes, whether an exception occurs or not.

## What is the difference between throw and throws?

`throw` is used to throw an exception, while `throws` is used in method signatures to declare that a method might throw an exception.

## How do I create a custom exception?

```java
public class CustomException extends Exception {
    public CustomException(String message) {
        super(message);
    }
}
```

## What is exception chaining?

Exception chaining allows you to wrap one exception inside another to preserve the original exception information.

## How do I use exception chaining?

```java
try {
    // some code
} catch (Exception e) {
    throw new CustomException("Custom message", e);
}
```

## What is multithreading in Java?

Multithreading is the ability of a program to execute multiple threads concurrently.

## What is a thread in Java?

A thread is a lightweight subprocess that represents the smallest unit of processing.

## How do I create a thread in Java?

```java
// Method 1: Extending Thread class
class MyThread extends Thread {
    public void run() {
        // thread code
    }
}

// Method 2: Implementing Runnable interface
class MyRunnable implements Runnable {
    public void run() {
        // thread code
    }
}
```

## How do I start a thread?

```java
MyThread thread = new MyThread();
thread.start();

// Or with Runnable
Thread thread = new Thread(new MyRunnable());
thread.start();
```

## What is the difference between start() and run()?

`start()` creates a new thread and calls `run()`, while `run()` executes in the current thread.

## What is thread synchronization?

Thread synchronization is the mechanism that ensures that only one thread can access a shared resource at a time.

## How do I synchronize methods?

```java
public synchronized void synchronizedMethod() {
    // synchronized code
}
```

## What is the synchronized block?

A synchronized block allows you to synchronize a specific block of code rather than an entire method.

## How do I use synchronized blocks?

```java
synchronized (object) {
    // synchronized code
}
```

## What is the volatile keyword?

The volatile keyword ensures that a variable's value is always read from main memory, not from thread cache.

## What are thread pools?

Thread pools are a collection of pre-initialized threads that can be reused to execute tasks.

## How do I create a thread pool?

```java
ExecutorService executor = Executors.newFixedThreadPool(5);
executor.submit(new MyRunnable());
executor.shutdown();
```
