# Rust Resources

Rust

- [The Rust Programming Language](https://doc.rust-lang.org/stable/book/)
- [Rust 程序设计语言 (zh)](https://rustwiki.org/zh-CN/book/title-page.html)
- https://rust-book.cs.brown.edu/
- [rust-by-example](https://doc.rust-lang.org/rust-by-example/)
- [The Cargo Book](https://doc.rust-lang.org/cargo/index.html)
- [The Little Book of Rust Macros](https://veykril.github.io/tlborm/introduction.html)
- [releases](https://releases.rs/)
- [github releases](https://github.com/rust-lang/rust/releases)

Rust OS

- [NUDT-OS-Book](https://flying-rind.github.io/mini-Rust-os/)
- [rcore-os](https://github.com/rcore-os)
- [rCore-Tutorial-Book-v3](https://rcore-os.cn/rCore-Tutorial-Book-v3/chapter0/5setup-devel-env.html)
- [haibo_chen](https://ipads.se.sjtu.edu.cn/pub/members/haibo_chen)

Others

- [一文读懂什么是进程、线程、协程](https://www.cnblogs.com/Survivalist/p/11527949.html)
- [rust-cli](https://rust-cli.github.io/book/index.html)

git submodule update --init --recursive

Commands

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

```
