# Rust Programming Language

## Ch9: 错误处理

Rust 要求你承认错误的可能性，并在你的代码编译前采取一些行动。

- 可恢复的 recoverable: 向用户报告错误
  - 使用 `Result` 处理
- 不可恢复的 unrecorverable: 立即 panic 停止程序
  - 使用 `panic!` 处理
  - 通过 backtrace 查看调用栈信息

### 使用 `Result` 处理可恢复的 (recoverable) 错误

```rust
use std::fs::File;

fn main() {
    let greeting_file_result = File::open("hello.txt");

    let greeting_file = match greeting_file_result {
        Ok(file) => file,
        Err(error) => panic!("Problem opening the file: {error:?}"),
    };
}
```

#### 匹配不同的错误

```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    let greeting_file_result = File::open("hello.txt");

    let greeting_file = match greeting_file_result {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("Problem creating the file: {e:?}"),
            },
            other_error => {
                panic!("Problem opening the file: {other_error:?}");
            }
        },
    };
}
```

使用`闭包`和 `unwrap_or_else` 方法

```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    let greeting_file = File::open("hello.txt").unwrap_or_else(|error| {
        if error.kind() == ErrorKind::NotFound {
            File::create("hello.txt").unwrap_or_else(|error| {
                panic!("Problem creating the file: {:?}", error);
            })
        } else {
            panic!("Problem opening the file: {:?}", error);
        }
    });
}
```

#### 失败时 panic 的简写：unwrap 和 expect

如果 Result 值是成员 Ok，unwrap 会返回 Ok 中的值。如果 Result 是成员 Err，unwrap 会为我们调用 panic!

```rust
use std::fs::File;

fn main() {
    let greeting_file = File::open("hello.txt").unwrap();
}
```

expect 与 unwrap 的使用方式一样：返回文件句柄或调用 panic! 宏。expect 在调用 panic! 时使用的错误信息将是我们传递给 expect 的参数，而不像 unwrap 那样使用默认的 panic! 信息。

```rust
use std::fs::File;

fn main() {
    let greeting_file = File::open("hello.txt")
        .expect("hello.txt should be included in this project");
}
```

#### 传播错误 propagating

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let username_file_result = File::open("hello.txt");

    let mut username_file = match username_file_result {
        Ok(file) => file,
        Err(e) => return Err(e),
    };

    let mut username = String::new();

    match username_file.read_to_string(&mut username) {
        Ok(_) => Ok(username),
        Err(e) => Err(e),
    }
}


```

#### 传播错误的简写：`?` 运算符

Result 值之后的 ? 被定义为与示例 9-6 中定义的处理 Result 值的 match 表达式有着完全相同的工作方式。如果 Result 的值是 Ok，这个表达式将会返回 Ok 中的值而程序将继续执行。如果值是 Err，Err 将作为整个函数的返回值，就好像使用了 return 关键字一样，这样错误值就被传播给了调用者。

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut username_file = File::open("hello.txt")?;
    let mut username = String::new();
    username_file.read_to_string(&mut username)?;
    Ok(username)
}
```

? 运算符所使用的错误值被传递给了 from 函数，它定义于标准库的 From trait 中，其用来将错误从一种类型转换为另一种类型。当 ? 运算符调用 from 函数时，收到的错误类型被转换为由当前函数返回类型所指定的错误类型。这在当函数返回单个错误类型来代表所有可能失败的方式时很有用，即使其可能会因很多种原因失败。

我们可以将示例 9-7 中的 read_username_from_file 函数修改为返回一个自定义的 OurError 错误类型。如果我们也定义了 impl From<io::Error> for OurError 来从 io::Error 构造一个 OurError 实例，那么 read_username_from_file 函数体中的 ? 运算符调用会调用 from 并转换错误而无需在函数中增加任何额外的代码。

我们甚至可以在 ? 之后直接使用链式方法调用来进一步缩短代码，如示例 9-8 所示：

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut username = String::new();

    File::open("hello.txt")?.read_to_string(&mut username)?;

    Ok(username)
}
```

more simplify:

```rust
use std::fs;
use std::io;

fn read_username_from_file() -> Result<String, io::Error> {
    fs::read_to_string("hello.txt")
}
```

在 `Option<T>` 值上使用 `?` 运算符

错误信息也提到 ? 也可用于 Option<T> 值。如同对 Result 使用 ? 一样，只能在返回 Option 的函数中对 Option 使用 ?。在 Option<T> 上调用 ? 运算符的行为与 Result<T, E> 类似：如果值是 None，此时 None 会从函数中提前返回。如果值是 Some，Some 中的值作为表达式的返回值同时函数继续。示例 9-11 中有一个从给定文本中返回第一行最后一个字符的函数的例子：

```rust
fn last_char_of_first_line(text: &str) -> Option<char> {
    text.lines().next()?.chars().last()
}
```

这个函数返回 Option<char> 因为它可能会在这个位置找到一个字符，也可能没有字符。这段代码获取 text 字符串 slice 作为参数并调用其 lines 方法，这会返回一个字符串中每一行的迭代器。因为函数希望检查第一行，所以调用了迭代器 next 来获取迭代器中第一个值。如果 text 是空字符串，next 调用会返回 None，此时我们可以使用 ? 来停止并从 last_char_of_first_line 返回 None。如果 text 不是空字符串，next 会返回一个包含 text 中第一行的字符串 slice 的 Some 值。

? 会提取这个字符串 slice，然后可以在字符串 slice 上调用 chars 来获取字符的迭代器。我们感兴趣的是第一行的最后一个字符，所以可以调用 last 来返回迭代器的最后一项。这是一个 Option，因为有可能第一行是一个空字符串，例如 text 以一个空行开头而后面的行有文本，像是 "\nhi"。不过，如果第一行有最后一个字符，它会返回在一个 Some 成员中。? 运算符作用于其中给了我们一个简洁的表达这种逻辑的方式。如果我们不能在 Option 上使用 ? 运算符，则不得不使用更多的方法调用或者 match 表达式来实现这些逻辑。

注意你可以在返回 Result 的函数中对 Result 使用 ? 运算符，可以在返回 Option 的函数中对 Option 使用 ? 运算符，但是不可以混合搭配。? 运算符不会自动将 Result 转化为 Option，反之亦然；在这些情况下，可以使用类似 Result 的 ok 方法或者 Option 的 ok_or 方法来显式转换。

幸运的是 main 函数也可以返回 `Result<(), E>`，示例 9-12 中的代码来自示例 9-10 不过修改了 main 的返回值为 `Result<(), Box<dyn Error>>` 并在结尾增加了一个 Ok(()) 作为返回值。这段代码可以编译：

```rust
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    let greeting_file = File::open("hello.txt")?;

    Ok(())
}
```

`Box<dyn Error>` 类型是一个 **trait 对象**（_trait object_）第十八章 [顾及不同类型值的 trait 对象”](https://kaisery.github.io/trpl-zh-cn/ch18-02-trait-objects.html#顾及不同类型值的-trait-对象) 部分会做介绍。目前可以将 `Box<dyn Error>` 理解为 “任何类型的错误”。在返回 `Box<dyn Error>` 错误类型 `main` 函数中对 `Result` 使用 `?` 是允许的，因为它允许任何 `Err` 值提前返回。即便 `main` 函数体从来只会返回 `std::io::Error` 错误类型，通过指定 `Box<dyn Error>`，这个签名也仍是正确的，甚至当 `main` 函数体中增加更多返回其他错误类型的代码时也是如此。

当 `main` 函数返回 `Result<(), E>`，如果 `main` 返回 `Ok(())` 可执行程序会以 `0` 值退出，而如果 `main` 返回 `Err` 值则会以非零值退出；成功退出的程序会返回整数 `0`，运行错误的程序会返回非 `0` 的整数。Rust 也会从二进制程序中返回与这个惯例相兼容的整数。

`main` 函数也可以返回任何实现了 [`std::process::Termination`](https://doc.rust-lang.org/std/process/trait.Termination.html)[ trait](https://doc.rust-lang.org/std/process/trait.Termination.html) 的类型，它包含了一个返回 `ExitCode` 的 `report` 函数。请查阅标准库文档了解更多为自定义类型实现 `Termination` trait 的细节。

现在我们讨论过了调用 `panic!` 或返回 `Result` 的细节，是时候回到它们各自适合哪些场景的话题了。

#### 创建自定义类型进行有效性验证

```rust
pub struct Guess {
    value: i32,
}

impl Guess {
    pub fn new(value: i32) -> Guess {
        if value < 1 || value > 100 {
            panic!("Guess value must be between 1 and 100, got {value}.");
        }

        Guess { value }
    }

    pub fn value(&self) -> i32 {
        self.value
    }
}

```

## Ch10: Generic Types, Traits, and lifetimes

- Generic Data Types
- Traits: Defining Shared Behavior
- Validating Referenvces with Liftimes

### Traits

- trait 定义了某个特定类型拥有可能与其他类型共享的功能。可以通过 trait 以一种抽象的方式定义共同行为。
- 可以使用 trait bounds 指定泛型是任何拥有特定行为的类型。

需要注意的限制是，只有在 trait 或类型至少有一个属于当前 crate 时，我们才能对类型实现该 trait。例如，可以为 `aggregator` crate 的自定义类型 `Tweet` 实现如标准库中的 `Display` trait，这是因为 `Tweet` 类型位于 `aggregator` crate 本地的作用域中。类似地，也可以在 `aggregator` crate 中为 `Vec<T>` 实现 `Summary`，这是因为 `Summary` trait 位于 `aggregator` crate 本地作用域中。

但是不能为外部类型实现外部 trait。例如，不能在 `aggregator` crate 中为 `Vec<T>` 实现 `Display` trait。这是因为 `Display` 和 `Vec<T>` 都定义于标准库中，它们并不位于 `aggregator` crate 本地作用域中。这个限制是被称为 **相干性**（_coherence_）的程序属性的一部分，或者更具体的说是 **孤儿规则**（_orphan rule_），其得名于不存在父类型。这条规则确保了其他人编写的代码不会破坏你代码，反之亦然。没有这条规则的话，两个 crate 可以分别对相同类型实现相同的 trait，而 Rust 将无从得知应该使用哪一个实现。

### Validating Referenvces with Liftimes: 使用生命周期确保引用有效

- 生命周期避免了悬垂引用 dangling reference
  - 借用检查器 borrow checker: 比较作用域以确保所有的借用都是有效的
- 函数中的泛型生命周期
- 生命周期注解语法
- 函数签名中的生命周期注解
- 深入理解生命周期
- 结构体定义中的生命周期注解
- 生命周期省略 Lifetime Elision
- 方法定义中的生命周期注解
- 静态生命周期
- 结合泛型类型参数、trait bounds 和生命周期
- 总结

## Ch11: Writing Automated Tests

- How to write
- Controlling How Tests Are Run
- Test Orgnizations

## Ch12: An I/O Project

## Ch13: Functional Language Features: Iterators and Closures

### Closures: Anonymous Functions that capture their Environment

### Iterators

## Ch14: More about Cargo and Creates.io

## Ch15: Smart Pointers

## Ch16: Fearless Concurrency

## Ch17: Object Oriented Programming Features in Rust

## Ch18: Patterns and Maching

## Ch19: Advanced Features

## Ch20: Building a Multithreaded WebServer
