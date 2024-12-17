# pest. The Elegant Parser

- [pest.rs](https://pest.rs/)
- [pest.docs.rs](https://docs.rs/pest/latest/pest/)
- [pest.book](https://pest.rs/book/)

pest is a general purpose parser written in Rust with a focus on **accessibility** , **correctness** , and **performance** . It uses [parsing expression grammars (or PEG)](https://en.wikipedia.org/wiki/Parsing_expression_grammar) as input, which are similar in spirit to regular expressions, but which offer the enhanced expressivity needed to parse complex languages.

pest 是一个用 Rust 编写的通用解析器，重点关注可访问性、正确性和性能。它使用解析表达式语法（或 PEG）作为输入，其本质上与正则表达式类似，但提供了解析复杂语言所需的增强表达能力。

```pest
alpha = { 'a'..'z' | 'A'..'Z' }
digit = { '0'..'9' }

ident = { (alpha | digit)+ }

ident_list = _{ !digit ~ ident ~ (" " ~ ident)+ }
          // ^
          // ident_list rule is silent (produces no tokens or error reports)
```

```c
int main() {
    return 5;
}
```

```txt
- FuncDecl
  - int_t: "int "
  - Identifier: "main"
  - FormalParams: ""
  - Block > Stmt > Return > Expr > Integer: "5"
```
