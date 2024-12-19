# Rust Resources

Rust

- [The Rust Reference](https://doc.rust-lang.org/reference/)
- [Rust Ref: zh](https://rustwiki.org/zh-CN/reference/)
- ~~
- [The Rust Programming Language](https://doc.rust-lang.org/stable/book/)
- [The Rust Programming Language: Experiment Type](https://rust-book.cs.brown.edu/)
- [Rust 程序设计语言 (zh) - 2022](https://rustwiki.org/zh-CN/book/title-page.html)
- [Rust 程序设计语言 (zh) - 2024-05-02 ](https://kaisery.github.io/trpl-zh-cn/)
- ~~
- [The Rustonomicon: The Dark Arts of Unsafe Rust](https://doc.rust-lang.org/stable/nomicon/)
- [Rust 语言圣经(Rust Course)](https://course.rs/about-book.html)
- [rust-by-example](https://doc.rust-lang.org/rust-by-example/)
- [The Cargo Book](https://doc.rust-lang.org/cargo/index.html)
- [The Little Book of Rust Macros](https://veykril.github.io/tlborm/introduction.html)
- ~~
- [Rust RFCs - RFC Book](https://rust-lang.github.io/rfcs/introduction.html)
- ~~
- [releases](https://releases.rs/)
- [github releases](https://github.com/rust-lang/rust/releases)

Rust Compiler

- [cranelift: a fast, secure, relatively simple and innovative compiler backend](https://cranelift.dev/)
- [rustc_codegen_cranelift](https://github.com/rust-lang/rustc_codegen_cranelift/)
- [wasmtime: About
  A lightweight WebAssembly runtime that is fast, secure, and standards-compliant](https://github.com/bytecodealliance/wasmtime/)

Rust OS

- [NUDT-OS-Book](https://flying-rind.github.io/mini-Rust-os/)
- [rcore-os](https://github.com/rcore-os)
- [rCore-Tutorial-Book-v3](https://rcore-os.cn/rCore-Tutorial-Book-v3/chapter0/5setup-devel-env.html)
- [haibo_chen](https://ipads.se.sjtu.edu.cn/pub/members/haibo_chen)

Others

- [一文读懂什么是进程、线程、协程](https://www.cnblogs.com/Survivalist/p/11527949.html)
- [rust-cli](https://rust-cli.github.io/book/index.html)

git submodule update --init --recursive

## Commands

```bash
# rustup: Install, manage, and update Rust toolchains.
rustup install/default/update/show

rustup self uninstall

# cargo: Rust's package manager and build system.
cargo new <project-name> # create a new Rust project

cargo build # build the current package
cargo run # build and run the current package
cargo check # check the current package for errors without building
cargo test # run the tests in the current package

cargo build --release # build the current package with optimizations

cargo doc --open # build all dependences doc and open in broswer
```

## OwnerShip

1. Rust 中的每一个值都有一个 **所有者** （ _owner_ ）。
2. 值在任一时刻有且只有一个所有者。
3. 当所有者（变量）离开作用域，这个值将被丢弃。

### 引用

- 在任意给定时间，**要么** 只能有一个可变引用，**要么** 只能有多个不可变引用。
- 引用必须总是有效的。

### Slice 类型

slice 允许你引用集合中一段连续的元素序列，而不用引用整个集合。slice 是一种引用，所以它没有所有权。

```rust
fn first_word(s: &String) -> &str {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }

    &s[..]
}

```

```rust
fn main() {
    let my_string = String::from("hello world");

    // `first_word` 适用于 `String`（的 slice），部分或全部
    let word = first_word(&my_string[0..6]);
    let word = first_word(&my_string[..]);
    // `first_word` 也适用于 `String` 的引用，
    // 这等价于整个 `String` 的 slice
    let word = first_word(&my_string);

    let my_string_literal = "hello world";

    // `first_word` 适用于字符串字面值，部分或全部
    let word = first_word(&my_string_literal[0..6]);
    let word = first_word(&my_string_literal[..]);

    // 因为字符串字面值已经 **是** 字符串 slice 了，
    // 这也是适用的，无需 slice 语法！
    let word = first_word(my_string_literal);
}
```

## Struct

#### 字段初始化简写语法（field init shorthand）

```rust
fn build_user(email: String, username: String) -> User {
    User {
        active: true,
        username,
        email,
        sign_in_count: 1,
    }
}
```

#### 结构体更新语法（struct update syntax）

使用旧实例的大部分值但改变其部分值来创建一个新的结构体实例

```rust
fn main() {
    // --snip--
    let user2 = User {
        email: String::from("another@example.com"),
        ..user1
    };
}
```

示例 5-7 中的代码也在 `user2` 中创建了一个新实例，但该实例中 `email` 字段的值与 `user1` 不同，而 `username`、 `active` 和 `sign_in_count` 字段的值与 `user1` 相同。`..user1` 必须放在最后，以指定其余的字段应从 `user1` 的相应字段中获取其值，但我们可以选择以任何顺序为任意字段指定值，而不用考虑结构体定义中字段的顺序。

请注意，结构更新语法就像带有 `=` 的赋值，因为它移动了数据，就像我们在[“变量与数据交互的方式（一）：移动”](https://kaisery.github.io/trpl-zh-cn/ch04-01-what-is-ownership.html#%E5%8F%98%E9%87%8F%E4%B8%8E%E6%95%B0%E6%8D%AE%E4%BA%A4%E4%BA%92%E7%9A%84%E6%96%B9%E5%BC%8F%E4%B8%80%E7%A7%BB%E5%8A%A8)部分讲到的一样。在这个例子中，总体上说我们在创建 `user2` 后就不能再使用 `user1` 了，因为 `user1` 的 `username` 字段中的 `String` 被移到 `user2` 中。如果我们给 `user2` 的 `email` 和 `username` 都赋予新的 `String` 值，从而只使用 `user1` 的 `active` 和 `sign_in_count` 值，那么 `user1` 在创建 `user2` 后仍然有效。`active` 和 `sign_in_count` 的类型是实现 `Copy` trait 的类型，所以我们在[“变量与数据交互的方式（二）：克隆”](https://kaisery.github.io/trpl-zh-cn/ch04-01-what-is-ownership.html#%E5%8F%98%E9%87%8F%E4%B8%8E%E6%95%B0%E6%8D%AE%E4%BA%A4%E4%BA%92%E7%9A%84%E6%96%B9%E5%BC%8F%E4%BA%8C%E5%85%8B%E9%9A%86) 部分讨论的行为同样适用。

#### 元组结构体（tuple structs）

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
}
```

#### 类单元结构体（unit-like structs）

- 没有任何字段的结构体
- 类单元结构体常常在你想要在某个类型上实现 trait 但不需要在类型中存储数据的时候发挥作用。

```rust
struct AlwaysEqual;

fn main() {
    let subject = AlwaysEqual;
}
```

sdsd

dsd
