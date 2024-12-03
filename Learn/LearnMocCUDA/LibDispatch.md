https://en.wikipedia.org/wiki/Grand_Central_Dispatch

[text](https://developer.apple.com/documentation/dispatch)

`libdispatch`, also known as **Grand Central Dispatch (GCD)**, is a library designed to provide efficient, system-level support for concurrent and parallel programming. It was first introduced by Apple as part of macOS and iOS but has since been made open-source and ported to other platforms like Linux.

---

## **What is `libdispatch` For?**

### 1. **Concurrency and Parallelism**
- `libdispatch` helps developers write concurrent programs more easily by managing threads and work queues internally.
- Instead of explicitly creating and managing threads, `libdispatch` allows you to schedule tasks (blocks of code) to execute asynchronously or synchronously.

### 2. **Efficient Resource Utilization**
- It uses a pool of threads and intelligently distributes tasks to available system resources, optimizing CPU and memory usage.
- Balances workloads across cores to maximize performance.

### 3. **Thread Safety**
- Simplifies thread-safe programming by encapsulating tasks in dispatch queues, avoiding race conditions and the need for extensive locks.

### 4. **Event Handling**
- Allows for event-driven programming by integrating tasks with file descriptors, timers, or other signals.

---

## **Key Features of `libdispatch`**

1. **Dispatch Queues**
   - A queue where tasks are scheduled to execute.
   - Types:
     - **Serial Queue**: Tasks are executed one at a time in order.
     - **Concurrent Queue**: Tasks are executed in parallel, as resources allow.
     - **Main Queue**: A special serial queue for tasks that must run on the main thread, typically UI updates.

2. **Work Items**
   - Blocks of code that you enqueue for execution.
   - Can be written in C, Objective-C, or Swift.

3. **Dispatch Sources**
   - Event sources that trigger the execution of tasks, such as timers, file descriptors, or custom triggers.

4. **Group Management**
   - Dispatch groups allow you to group tasks and wait for their collective completion.

5. **Timers**
   - Built-in support for creating and managing timers in a queue-based system.

---

## **When to Use `libdispatch`**

1. **Asynchronous Tasks**
   - When tasks can be performed independently of each other, allowing the program to remain responsive (e.g., downloading a file while processing user input).

2. **Parallel Processing**
   - Tasks that can run simultaneously to leverage multicore processors, such as image processing, data analysis, or sorting algorithms.

3. **Thread-Safe Code**
   - Simplifies concurrent access to shared resources without explicit locks.

4. **Event-Driven Programming**
   - Ideal for applications that need to respond to events like network requests, user input, or timers.

5. **Performance-Critical Applications**
   - Helps offload heavy computational tasks without blocking the main thread.

---

## **How to Use `libdispatch`**

### In C:
```c
#include <dispatch/dispatch.h>

void my_function() {
    printf("Task executed\n");
}

int main() {
    // Create a dispatch queue
    dispatch_queue_t queue = dispatch_queue_create("com.example.myqueue", DISPATCH_QUEUE_CONCURRENT);

    // Add tasks to the queue
    dispatch_async(queue, ^{
        my_function();
    });

    dispatch_async(queue, ^{
        printf("Another task executed\n");
    });

    // Wait for some time to let tasks finish
    sleep(1);

    // Clean up
    dispatch_release(queue);
    return 0;
}
```

### In Swift (if using on macOS or iOS):
```swift
import Dispatch

let queue = DispatchQueue(label: "com.example.myqueue", attributes: .concurrent)

// Schedule tasks
queue.async {
    print("Task 1 executed")
}

queue.async {
    print("Task 2 executed")
}

// Dispatch to the main queue
DispatchQueue.main.async {
    print("UI updates can go here")
}
```

---

## **Advantages of `libdispatch`**
1. **Simplifies Multithreading**: Abstracts low-level threading details.
2. **Efficient**: Optimizes resource use, particularly in multicore systems.
3. **Portable**: Works on macOS, iOS, Linux, and more.

---

## **Limitations**
- Not a substitute for more complex threading needs like thread-local storage or thread prioritization.
- For very fine-grained control of threads, alternatives like POSIX threads or C++ `std::thread` might be better.

---

In short, `libdispatch` is an essential tool for building modern, high-performance applications, especially when concurrency and responsiveness are crucial.

## macOS GCD - Grand Central Dispatc

[macOS GCD - Grand Central Dispatch](https://book.hacktricks.xyz/macos-hardening/macos-security-and-privilege-escalation/macos-gcd-grand-central-dispatch)

Grand Central Dispatch (GCD), also known as libdispatch (libdispatch.dyld), is available in both macOS and iOS. It's a technology developed by Apple to optimize application support for concurrent (multithreaded) execution on multicore hardware.
Grand Central Dispatch (GCD)，也称为libdispatch ( libdispatch.dyld )，在 macOS 和 iOS 中均可用。它是 Apple 开发的一项技术，旨在优化多核硬件上并发（多线程）执行的应用程序支持。

GCD provides and manages FIFO queues to which your application can submit tasks in the form of block objects. Blocks submitted to dispatch queues are executed on a pool of threads fully managed by the system. GCD automatically creates threads for executing the tasks in the dispatch queues and schedules those tasks to run on the available cores.
GCD提供并管理FIFO 队列，您的应用程序可以向其中以块对象的形式提交任务。提交到调度队列的块在完全由系统管理的线程池上执行。 GCD 自动创建线程来执行调度队列中的任务，并安排这些任务在可用内核上运行。

In summary, to execute code in parallel, processes can send blocks of code to GCD, which will take care of their execution. Therefore, processes don't create new threads; GCD executes the given code with its own pool of threads (which might increase or decrease as necessary).
总之，要并行执行代码，进程可以将代码块发送到 GCD ，GCD 将负责它们的执行。因此，进程不会创建新线程；而是创建新线程。 GCD 使用自己的线程池执行给定的代码（可能会根据需要增加或减少）。

This is very helpful to manage parallel execution successfully, greatly reducing the number of threads processes create and optimising the parallel execution. This is ideal for tasks that require great parallelism (brute-forcing?) or for tasks that shouldn't block the main thread: For example, the main thread on iOS handles UI interactions, so any other functionality that could make the app hang (searching, accessing a web, reading a file...) is managed this way.
这对于成功管理并行执行非常有帮助，大大减少了进程创建的线程数量并优化了并行执行。这非常适合需要大量并行性（强制？）的任务或不应阻塞主线程的任务：例如，iOS 上的主线程处理 UI 交互，因此任何其他可能导致应用程序挂起的功能（搜索、访问网络、读取文件...）都是以这种方式进行管理的。

[GCD Part 1: Queues and methods](https://shchukin-alex.medium.com/gcd-queues-and-methods-f12453f529e7)

[Grand Central Dispatch (GCD) on FreeBSD](https://wiki.freebsd.org/GrandCentralDispatch)

[GCD Handbook](https://khanlou.com/2016/04/the-GCD-handbook/)