# Java Collections and Data Structures FAQ

## What are Collections in Java?

Collections are data structures that store and manipulate groups of objects. The Java Collections Framework provides a unified architecture for representing and manipulating collections.

## What is the difference between Collection and Collections?

Collection is an interface that represents a group of objects, while Collections is a utility class that provides static methods for manipulating collections.

## What are the main interfaces in the Collections Framework?

The main interfaces are Collection, List, Set, Queue, and Map.

## What is a List in Java?

A List is an ordered collection that allows duplicate elements. It maintains insertion order.

## What are the main List implementations?

ArrayList, LinkedList, and Vector are the main List implementations.

## What is the difference between ArrayList and LinkedList?

ArrayList is backed by an array and provides fast random access, while LinkedList is backed by a doubly-linked list and provides fast insertion/deletion.

## How do I create an ArrayList?

```java
List<String> list = new ArrayList<>();
list.add("Hello");
list.add("World");
```

## What is a Set in Java?

A Set is a collection that does not allow duplicate elements.

## What are the main Set implementations?

HashSet, LinkedHashSet, and TreeSet are the main Set implementations.

## What is the difference between HashSet and TreeSet?

HashSet provides O(1) average time complexity but no ordering, while TreeSet provides O(log n) time complexity with sorted ordering.

## How do I create a HashSet?

```java
Set<String> set = new HashSet<>();
set.add("Apple");
set.add("Banana");
```

## What is a Map in Java?

A Map is an object that maps keys to values. It cannot contain duplicate keys.

## What are the main Map implementations?

HashMap, LinkedHashMap, TreeMap, and Hashtable are the main Map implementations.

## What is the difference between HashMap and TreeMap?

HashMap provides O(1) average time complexity but no ordering, while TreeMap provides O(log n) time complexity with sorted ordering by keys.

## How do I create a HashMap?

```java
Map<String, Integer> map = new HashMap<>();
map.put("Apple", 1);
map.put("Banana", 2);
```

## What is a Queue in Java?

A Queue is a collection designed for holding elements prior to processing. It follows FIFO (First In, First Out) principle.

## What are the main Queue implementations?

LinkedList, PriorityQueue, and ArrayDeque are the main Queue implementations.

## How do I use a Queue?

```java
Queue<String> queue = new LinkedList<>();
queue.offer("First");
queue.offer("Second");
String first = queue.poll(); // removes and returns "First"
```

## What is an Iterator?

An Iterator is an object that enables you to traverse through a collection and remove elements during iteration.

## How do I use an Iterator?

```java
List<String> list = Arrays.asList("A", "B", "C");
Iterator<String> iterator = list.iterator();
while (iterator.hasNext()) {
    System.out.println(iterator.next());
}
```

## What is the difference between Iterator and ListIterator?

ListIterator extends Iterator and provides bidirectional traversal and modification capabilities.

## What are generics in Java?

Generics enable types (classes and interfaces) to be parameters when defining classes, interfaces, and methods.

## How do I use generics with collections?

```java
List<String> stringList = new ArrayList<>();
Map<Integer, String> numberMap = new HashMap<>();
```
