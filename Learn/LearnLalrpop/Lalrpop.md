# Lalrpop

LR(1) parser generator for Rust.

[LALRPOP  拉尔波普](https://lalrpop.github.io/lalrpop/#lalrpop)

LALRPOP is a parser generator, similar in principle to [YACC](http://dinosaur.compilertools.net/yacc/), [ANTLR](http://www.antlr.org/), [Menhir](http://gallium.inria.fr/~fpottier/menhir/), and other such programs. In general, it has the grand ambition of being the most usable parser generator ever. This ambition is most certainly not fully realized: right now, it's fairly standard, maybe even a bit subpar in some areas. But hey, it's young. For the most part, this README is intended to describe the current behavior of LALRPOP, but in some places it includes notes for planned future changes.

LALRPOP 是一个解析器生成器，原理类似于  YACC、ANTLR、Menhir  和其他此类程序。总的来说，它有一个宏伟的雄心壮志，那就是成为有史以来最有用的解析器生成器。这个雄心壮志肯定没有完全实现：现在，它相当标准，甚至在某些领域可能有点低于标准。但是，嘿，它很年轻。在大多数情况下，本 README 旨在描述 LALRPOP 的当前行为，但在某些地方，它包括计划的未来更改的说明。

1. add lalrpop to rust project
2. define grammar in lalrpop file
   1. define asts, build ast
   2. use macros
   3. use errors and error recovery
   4. pass param to parser

## Grammar Example

```rust
// Term: Num | '(' Term ')'
pub Term: i32 = {
    <n: Num> => n,
    "(" <t:Term> ")" => t,
}
// Num: r"[0-9]+"
Num: i32 = <s:r"[0-9]+"> => i32::from_str(&s).unwrap();
```

Precedence and associativity can be specified using attributes:

```rust
pub Expr: i32 = {
    #[precedence(level="0")] // Highest precedence
    Term,
    #[precedence(level="1")] #[assoc(side="left")]
    <l:Expr> "*" <r:Expr> => l * r,
    <l:Expr> "/" <r:Expr> => l / r,
    #[precedence(level="2")] #[assoc(side="left")]
    <l:Expr> "+" <r:Expr> => l + r,
    <l:Expr> "-" <r:Expr> => l - r,
};
```

Use AST to represent the parsed tree:

```rust
// ast.rs
use std::fmt::{Debug, Error, Formatter};

pub enum Expr {
    Number(i32),
    Op(Box<Expr>, Opcode, Box<Expr>),
    Error,
}

pub enum ExprSymbol<'input> {
    NumSymbol(&'input str),
    Op(Box<ExprSymbol<'input>>, Opcode, Box<ExprSymbol<'input>>),
    Error,
}

#[derive(Copy, Clone)]
pub enum Opcode {
    Mul,
    Div,
    Add,
    Sub,
}

impl Debug for Expr {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), Error> {
        use self::Expr::*;
        match *self {
            Number(n) => write!(fmt, "{:?}", n),
            Op(ref l, op, ref r) => write!(fmt, "({:?} {:?} {:?})", l, op, r),
            Error => write!(fmt, "error"),
        }
    }
}

impl Debug for ExprSymbol<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), Error> {
        use self::ExprSymbol::*;
        match *self {
            NumSymbol(n) => write!(fmt, "{:?}", n),
            Op(ref l, op, ref r) => write!(fmt, "({:?} {:?} {:?})", l, op, r),
            Error => write!(fmt, "error"),
        }
    }
}

impl Debug for Opcode {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), Error> {
        use self::Opcode::*;
        match *self {
            Mul => write!(fmt, "*"),
            Div => write!(fmt, "/"),
            Add => write!(fmt, "+"),
            Sub => write!(fmt, "-"),
        }
    }
}

```

```rust
// grammar.lalrpop
use std::str::FromStr;
use crate::ast::{Expr, Opcode};

grammar;

pub Expr: Box<Expr> = { // (1)
    Expr ExprOp Factor => Box::new(Expr::Op(<>)), // (2)
    Factor,
};

ExprOp: Opcode = { // (3)
    "+" => Opcode::Add,
    "-" => Opcode::Sub,
};

Factor: Box<Expr> = {
    Factor FactorOp Term => Box::new(Expr::Op(<>)),
    Term,
};

FactorOp: Opcode = {
    "*" => Opcode::Mul,
    "/" => Opcode::Div,
};

Term: Box<Expr> = {
    Num => Box::new(Expr::Number(<>)),
    "(" <Expr> ")"
};

Num: i32 = {
    r"[0-9]+" => i32::from_str(<>).unwrap()
};
```

## Lexer

```txt
   Num: i32 = r"[0-9]+" => i32::from_str(<>).unwrap();
// ~~~  ~~~   ~~~~~~~~~    ~~~~~~~~~~~~~~~~~~~~~~~~~~
// |    |     |                Action code
// |    |     Symbol(s) that should match
// |    Return type
// Name of nonterminal
```

```txt
        +-------------------+    +---------------------+
Text -> | Lexer             | -> | Parser              |
        |                   |    |                     |
        | Applies regex to  |    | Consumes terminals, |
        | produce terminals |    | executes your code  |
        +-------------------+    | as it recognizes    |
                                 | nonterminals        |
                                 +---------------------+

```

词法分析优先级问题：使用 match 显式指定优先级。

Simple match declarations
简单 match 声明

A match declaration lets you explicitly give the precedence between terminals. In its simplest form, it consists of just ordering regular expressions and string literals into groups, with the higher precedence items coming first. So, for example, we could resolve our conflict above by giving r"[0-9]+" precedence over r"\w+", thus saying that if something can be lexed as a number, we'll do that, and otherwise consider it to be an identifier.

match 声明允许您显式指定终端之间的优先级。在最简单的形式中，它只包括将正则表达式和字符串文本排序到组中，优先级较高的项排在最前面。因此，例如，我们可以通过赋予 r“[0-9]+“优先于 r”\w+“ 来解决上面的冲突，从而表示如果某个东西可以被解释为一个数字，我们就会这样做，否则就将其视为标识符。

```rust
match {
    r"[0-9]+"
} else {
    r"\w+",
    _
}
```

The final `_` indicates that other string literals and regular expressions that appear elsewhere in the grammar (e.g., "(" or "22") should be added into that final level of precedence (without an `_`, it is illegal to use a terminal that does not appear in the match declaration).

结尾的 `_` 表示出现在语法中其他位置的字符串文字和正则表达式（例如，“（” 或 “22”）应该添加到最终的优先级中（如果没有 `_`，使用未出现在 match 声明中的终结符是非法的）。

```rust
// fixed literals get precedence over regular expressions
match {
    r"[0-9]+",
    "22"
} else {
    r"\w+",
    _
}
```

使用 match 声明为正则表达式命名，这样我们就不必直接在语法中输入它们。

```rust
match {
    r"[0-9]+",
    "22"
} else {
    r"\w+" => ID, // <-- give a name here
    _
}
```

```rust
pub Term = {
    Num,
    "(" <Term> ")",
    "22" => "Twenty-two!".to_string(),
    ID => format!("Id({})", <>), // <-- changed this
};
```

Customizing skipping between tokens
自定义在标记之间跳过（用于跳过注释内容）

```rust
match {
    r"\s*" => { }, // The default whitespace skipping is disabled if an `ignore pattern` is specified
    r"//[^\n\r]*[\n\r]*" => { }, // Skip `// comments`
    r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/" => { },  // Skip `/* comments */`
}

```
