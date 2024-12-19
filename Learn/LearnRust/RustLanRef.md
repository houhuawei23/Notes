# Rust Language Reference

## Ref

### Traits 特性

```rust
unsafe? trait IDENTIFIER  GenericParams? ( : TypeParamBounds? )? WhereClause? {
  InnerAttribute*
  AssociatedItem*
}
```

- A `trait` describes an abstract interface that types can implement.
- This interface consists of `associated items`, which come in three varieties.
- Trait declaration defines a trait in the `type namespace` of the module or block where it is located.

  - `Associated items` are defined as members of the trait within their respective namespaces.
  - `Associated types` are defined in the type namespace.
  - `Associated constants` and `associated functions` are defined in the value namespace.

- All traits define an implicit type parameter `Self` that refers to “the type that is implementing this interface”.
- Traits may also contain additional type parameters. These type parameters, including `Self`, may be constrained by other traits and so forth as usual.
- Traits are implemented for specific types through separate [implementations](https://doc.rust-lang.org/reference/items/implementations.html).
- Trait functions may omit the function body by replacing it with a semicolon. This indicates that the implementation must define the function.
- If the trait function defines a body, this definition acts as a default for any implementation which does not override it. Similarly, associated constants may omit the equals sign and expression to indicate implementations must define the constant value. Associated types must never define the type, the type may only be specified in an implementation.
- `Trait functions` are not allowed to be `const`.

---

- `trait` 描述了一个类能实现的抽象接口。
- 这个 `接口`由一些 `相关项目`构成，包括：
  - functions 函数
  - types 类型
  - constants 常数
- trait 的声明在该声明所在的 module 或 block 中的命名空间定义了一个 trait
- 所有特征都定义了一个隐式类型参数 Self ，它指的是“实现该接口的类型”。特征还可能包含其他类型参数。这些类型参数，包括 Self ，可能会像往常一样受到其他特征等的约束。
- 对一个特定类型的 trait 实现是与 trait 的定义相分开的
- 特征函数可以通过用分号替换来省略函数体。这表明实现必须定义该函数。
- 如果特征函数定义了一个函数体，则此定义将充当任何不覆盖它的实现的默认值。
- 类似地，关联的常量可以省略等号和表达式以指示实现必须定义常量值。
- 关联类型绝不能定义该类型，该类型只能在实现中指定。
- trait function 不能为 const ？

```rust
// Examples of associated trait items with and without definitions.
trait Example {
    const CONST_NO_DEFAULT: i32;
    const CONST_WITH_DEFAULT: i32 = 99;
    type TypeNoDefault;
    fn method_without_default(&self);
    fn method_with_default(&self) {}
}
```

#### Trait bounds

`泛型`项可能会使用 traits 作为 type parameters 的限制（bounds）。

### Trait and lifetime bounds

[Trait](https://doc.rust-lang.org/reference/items/traits.html#trait-bounds) and lifetime bounds provide a way for [generic items](https://doc.rust-lang.org/reference/items/generics.html) to restrict which types and lifetimes are used as their parameters. Bounds can be provided on any type in a [where clause](https://doc.rust-lang.org/reference/items/generics.html#where-clauses). There are also shorter forms for certain common cases:

特征和生命周期界限为通用项提供了一种方法来限制将哪些类型和生命周期用作其参数。可以在 where 子句中为任何类型提供界限。对于某些常见情况，还有更简短的形式：

- Bounds written after declaring a [generic parameter](https://doc.rust-lang.org/reference/items/generics.html): `fn f<A: Copy>() {}` is the same as `fn f<A>() where A: Copy {}`.
- 声明通用参数后写入的界限： fn f<A: Copy>() {}与 fn f `<A>`() where A: Copy {} 。
- In trait declarations as [supertraits](https://doc.rust-lang.org/reference/items/traits.html#supertraits): `trait Circle : Shape {}` is equivalent to `trait Circle where Self : Shape {}`.
- 在作为 supertraits 的特征声明中： trait Circle : Shape {}相当于  trait Circle where Self : Shape {} 。
- In trait declarations as bounds on [associated types](https://doc.rust-lang.org/reference/items/associated-items.html#associated-types): `trait A { type B: Copy; }` is equivalent to `trait A where Self::B: Copy { type B; }`.
- 在特征声明中作为关联类型的边界： trait A { type B: Copy; }相当于  trait A where Self::B: Copy { type B; } 。

Bounds on an item must be satisfied when using the item. When type checking and borrow checking a generic item, the bounds can be used to determine that a trait is implemented for a type. For example, given `Ty: Trait`

使用物品时必须满足物品的限制。当类型检查和借用检查通用项时，边界可用于确定是否为类型实现了特征。例如，给定 Ty: Trait

- In the body of a generic function, methods from `Trait` can be called on `Ty` values. Likewise associated constants on the `Trait` can be used.
  - 在泛型函数体内，可以在 Ty 上调用 Trait 中的方法 价值观。同样可以使用 Trait 上的关联常量。
- Associated types from `Trait` can be used.
  - 可以使用 Trait 中的关联类型。
- Generic functions and types with a `T: Trait` bounds can be used with `Ty` being used for `T`.
  - 带有 T: Trait 边界可以与 Ty 一起使用 用于 T 。

### Trait objects 特性对象

A `trait object` is an opaque value of another type that implements a set of traits. The set of traits is made up of an [object safe](https://doc.rust-lang.org/reference/items/traits.html#object-safety) `base trait` plus any number of [auto traits](https://doc.rust-lang.org/reference/special-types-and-traits.html#auto-traits).

特征对象是实现一组特征的另一种类型的不透明值。该特征集由 `对象安全基本特征(object safe base trait)`加上任意数量的 `自动特征 (auto traits)`组成。

Trait objects implement the `base trait`, its `auto traits`, and any [supertraits](https://doc.rust-lang.org/reference/items/traits.html#supertraits) of the base trait.

Trait 对象实现基本特征、其自动特征和任何基本特征的超级特征。

Trait objects are written as the keyword `dyn` followed by a set of `trait bounds`, but with the following restrictions on the trait bounds.

All traits except the first trait must be auto traits, there may not be more than one lifetime, and opt-out bounds (e.g. `?Sized`) are not allowed. Furthermore, paths to traits may be parenthesized.

Trait 对象被编写为关键字 dyn 后跟一组 `特征边界`，但对特征边界有以下限制。

除第一个特征之外的所有特征都必须是自动特征，不能有超过一个生命周期，并且不允许选择退出边界（例如?Sized ）。

此外，特征的路径可以用括号括起来。

For example, given a trait `Trait`, the following are all trait objects:例如，给定一个特征 Trait ，以下都是特征对象：

- `dyn Trait`
- `dyn Trait + Send`
- `dyn Trait + Send + Sync`
- `dyn Trait + 'static`
- `dyn Trait + Send + 'static`
- `dyn Trait +`
- `dyn 'static + Trait`.
- `dyn (Trait)`

Two trait object types alias each other if the base traits alias each other and if the sets of auto traits are the same and the lifetime bounds are the same. For example, `dyn Trait + Send + UnwindSafe` is the same as `dyn Trait + UnwindSafe + Send`.

Due to the opaqueness of which concrete type the value is of, trait objects are [dynamically sized types](https://doc.rust-lang.org/reference/dynamically-sized-types.html). Like all DSTs, trait objects are used behind some type of pointer; for example `&dyn SomeTrait` or `Box<dyn SomeTrait>`. Each instance of a pointer to a trait object includes:

由于值属于哪种具体类型的不透明性，特征对象是 [动态调整大小的类型](https://doc.rust-lang.org/reference/dynamically-sized-types.html)。像所有 DSTs 一样 ，特征对象用在某种类型的指针后面；例如 `&dyn SomeTrait`或 `Box<dyn SomeTrait>` 。指向特征对象的指针的每个实例包括：

- a pointer to an instance of a type `T` that implements `SomeTrait`
  - 指向实现 `SomeTrait`的类型 `T`实例的指针
- a _virtual method table_ , often just called a _vtable_ , which contains, for each method of `SomeTrait` and its [supertraits](https://doc.rust-lang.org/reference/items/traits.html#supertraits) that `T` implements, a pointer to `T`’s implementation (i.e. a function pointer).
  - 虚拟方法表，通常简称为 vtable ，其中包含 T 实现的 SomeTrait 及其超特征的每个方法，一个指向 T 实现的指针（即函数指针）。

The purpose of trait objects is to permit “late binding” of methods. Calling a method on a trait object results in virtual dispatch at runtime: that is, a function pointer is loaded from the trait object vtable and invoked indirectly. The actual implementation for each vtable entry can vary on an object-by-object basis.

特征对象的目的是允许方法的“后期绑定”。在特征对象上调用方法会导致运行时虚拟分派：也就是说，从特征对象 vtable 加载函数指针并间接调用。每个 vtable 条目的实际实现可能因对象而异。

An example of a trait object:

```rust
trait Printable {
    fn stringify(&self) -> String;
}

impl Printable for i32 {
    fn stringify(&self) -> String { self.to_string() }
}

fn print(a: Box<dyn Printable>) {
    println!("{}", a.stringify());
}

fn main() {
    print(Box::new(10) as Box<dyn Printable>);
}
```

In this example, the trait Printable occurs as a trait object in both the type signature of print, and the cast expression in main.

在此示例中，特征 Printable 作为特征对象出现在 print 的类型签名和 main 中的强制转换表达式中。

Trait Object Lifetime Bounds

特征对象生命周期界限

Since a trait object can contain references, the lifetimes of those references need to be expressed as part of the trait object. This lifetime is written as `Trait + 'a`. There are `defaults` that allow this lifetime to usually be inferred with a sensible choice.

由于特征对象可以包含引用，因此这些引用的生命周期 需要表达为特征对象的一部分。此 lifetime 可写为 `Trait + 'a` .有一些默认值允许通常通过明智的选择来推断此生命周期。

## Attributes 属性

- [attributes](https://doc.rust-lang.org/reference/attributes.html)

```rust
// inner attribute
#![Attr]
// outer attribute
#[Arrt]
```

An `attribute` is a general, free-form metadatum that is interpreted according to name, convention, language, and compiler version. Attributes are modeled on Attributes in [ECMA-335](https://www.ecma-international.org/publications-and-standards/standards/ecma-335/), with the syntax coming from [ECMA-334](https://www.ecma-international.org/publications-and-standards/standards/ecma-334/) (C#).

_Inner attributes_ , written with a bang (`!`) after the hash (`#`), apply to the item that the attribute is declared within.

- _Outer attributes_ , written without the bang after the hash, apply to the thing that follows the attribute.

The attribute consists of a path to the attribute, followed by an optional delimited token tree whose interpretation is defined by the attribute. Attributes other than macro attributes also allow the input to be an equals sign (`=`) followed by an expression. See the [meta item syntax](https://doc.rust-lang.org/reference/attributes.html#meta-item-attribute-syntax) below for more details.

An attribute may be unsafe to apply. To avoid undefined behavior when using these attributes, certain obligations that cannot be checked by the compiler must be met. To assert these have been, the attribute is wrapped in `unsafe(..)`, e.g. `#[unsafe(no_mangle)]`.

The following attributes are unsafe:

- [`export_name`](https://doc.rust-lang.org/reference/abi.html#the-export_name-attribute)
- [`link_section`](https://doc.rust-lang.org/reference/abi.html#the-link_section-attribute)
- [`no_mangle`](https://doc.rust-lang.org/reference/abi.html#the-no_mangle-attribute)

Attributes can be classified into the following kinds:

- [Built-in attributes](https://doc.rust-lang.org/reference/attributes.html#built-in-attributes-index)
- [Macro attributes](https://doc.rust-lang.org/reference/procedural-macros.html#attribute-macros)
- [Derive macro helper attributes](https://doc.rust-lang.org/reference/procedural-macros.html#derive-macro-helper-attributes)
- [Tool attributes](https://doc.rust-lang.org/reference/attributes.html#tool-attributes)

Attributes may be applied to many things in the language:

- All [item declarations](https://doc.rust-lang.org/reference/items.html) accept outer attributes while [external blocks](https://doc.rust-lang.org/reference/items/external-blocks.html), [functions](https://doc.rust-lang.org/reference/items/functions.html), [implementations](https://doc.rust-lang.org/reference/items/implementations.html), and [modules](https://doc.rust-lang.org/reference/items/modules.html) accept inner attributes.
- Most [statements](https://doc.rust-lang.org/reference/statements.html) accept outer attributes (see [Expression Attributes](https://doc.rust-lang.org/reference/expressions.html#expression-attributes) for limitations on expression statements).
- [Block expressions](https://doc.rust-lang.org/reference/expressions/block-expr.html) accept outer and inner attributes, but only when they are the outer expression of an [expression statement](https://doc.rust-lang.org/reference/statements.html#expression-statements) or the final expression of another block expression.
- [Enum](https://doc.rust-lang.org/reference/items/enumerations.html) variants and [struct](https://doc.rust-lang.org/reference/items/structs.html) and [union](https://doc.rust-lang.org/reference/items/unions.html) fields accept outer attributes.
- [Match expression arms](https://doc.rust-lang.org/reference/expressions/match-expr.html) accept outer attributes.
- [Generic lifetime or type parameter](https://doc.rust-lang.org/reference/items/generics.html) accept outer attributes.
- Expressions accept outer attributes in limited situations, see [Expression Attributes](https://doc.rust-lang.org/reference/expressions.html#expression-attributes) for details.
- [Function](https://doc.rust-lang.org/reference/items/functions.html), [closure](https://doc.rust-lang.org/reference/expressions/closure-expr.html) and [function pointer](https://doc.rust-lang.org/reference/types/function-pointer.html) parameters accept outer attributes. This includes attributes on variadic parameters denoted with `...` in function pointers and [external blocks](https://doc.rust-lang.org/reference/items/external-blocks.html#variadic-functions).

Examples:

```rust
// General metadata applied to the enclosing module or crate.
#![crate_type = "lib"]

// A function marked as a unit test
#[test]
fn test_foo() {
    /* ... */
}

// A conditionally-compiled module
#[cfg(target_os = "linux")]
mod bar {
    /* ... */
}

// A lint attribute used to suppress a warning/error
#[allow(non_camel_case_types)]
type int8_t = i8;

// Inner attribute applies to the entire function.
fn some_unused_variables() {
  #![allow(unused_variables)]

  let x = ();
  let y = ();
  let z = ();
}

```

## [Built-in attributes index](https://doc.rust-lang.org/reference/attributes.html#built-in-attributes-index)

The following is an index of all built-in attributes.

- Conditional compilation
  - [`cfg`](https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg-attribute) — Controls conditional compilation.
  - [`cfg_attr`](https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg_attr-attribute) — Conditionally includes attributes.
- Testing
  - [`test`](https://doc.rust-lang.org/reference/attributes/testing.html#the-test-attribute) — Marks a function as a test.
  - [`ignore`](https://doc.rust-lang.org/reference/attributes/testing.html#the-ignore-attribute) — Disables a test function.
  - [`should_panic`](https://doc.rust-lang.org/reference/attributes/testing.html#the-should_panic-attribute) — Indicates a test should generate a panic.
- Derive
  - [`derive`](https://doc.rust-lang.org/reference/attributes/derive.html) — Automatic trait implementations.
  - [`automatically_derived`](https://doc.rust-lang.org/reference/attributes/derive.html#the-automatically_derived-attribute) — Marker for implementations created by `derive`.
- Macros
  - [`macro_export`](https://doc.rust-lang.org/reference/macros-by-example.html#path-based-scope) — Exports a `macro_rules` macro for cross-crate usage.
  - [`macro_use`](https://doc.rust-lang.org/reference/macros-by-example.html#the-macro_use-attribute) — Expands macro visibility, or imports macros from other crates.
  - [`proc_macro`](https://doc.rust-lang.org/reference/procedural-macros.html#function-like-procedural-macros) — Defines a function-like macro.
  - [`proc_macro_derive`](https://doc.rust-lang.org/reference/procedural-macros.html#derive-macros) — Defines a derive macro.
  - [`proc_macro_attribute`](https://doc.rust-lang.org/reference/procedural-macros.html#attribute-macros) — Defines an attribute macro.
- Diagnostics
  - [`allow`](https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes), [`expect`](https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes), [`warn`](https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes), [`deny`](https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes), [`forbid`](https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes) — Alters the default lint level.
  - [`deprecated`](https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-deprecated-attribute) — Generates deprecation notices.
  - [`must_use`](https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute) — Generates a lint for unused values.
  - [`diagnostic::on_unimplemented`](https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-diagnosticon_unimplemented-attribute) — Hints the compiler to emit a certain error message if a trait is not implemented.
- ABI, linking, symbols, and FFI
  - [`link`](https://doc.rust-lang.org/reference/items/external-blocks.html#the-link-attribute) — Specifies a native library to link with an `extern` block.
  - [`link_name`](https://doc.rust-lang.org/reference/items/external-blocks.html#the-link_name-attribute) — Specifies the name of the symbol for functions or statics in an `extern` block.
  - [`link_ordinal`](https://doc.rust-lang.org/reference/items/external-blocks.html#the-link_ordinal-attribute) — Specifies the ordinal of the symbol for functions or statics in an `extern` block.
  - [`no_link`](https://doc.rust-lang.org/reference/items/extern-crates.html#the-no_link-attribute) — Prevents linking an extern crate.
  - [`repr`](https://doc.rust-lang.org/reference/type-layout.html#representations) — Controls type layout.
  - [`crate_type`](https://doc.rust-lang.org/reference/linkage.html) — Specifies the type of crate (library, executable, etc.).
  - [`no_main`](https://doc.rust-lang.org/reference/crates-and-source-files.html#the-no_main-attribute) — Disables emitting the `main` symbol.
  - [`export_name`](https://doc.rust-lang.org/reference/abi.html#the-export_name-attribute) — Specifies the exported symbol name for a function or static.
  - [`link_section`](https://doc.rust-lang.org/reference/abi.html#the-link_section-attribute) — Specifies the section of an object file to use for a function or static.
  - [`no_mangle`](https://doc.rust-lang.org/reference/abi.html#the-no_mangle-attribute) — Disables symbol name encoding.
  - [`used`](https://doc.rust-lang.org/reference/abi.html#the-used-attribute) — Forces the compiler to keep a static item in the output object file.
  - [`crate_name`](https://doc.rust-lang.org/reference/crates-and-source-files.html#the-crate_name-attribute) — Specifies the crate name.
- Code generation
  - [`inline`](https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute) — Hint to inline code.
  - [`cold`](https://doc.rust-lang.org/reference/attributes/codegen.html#the-cold-attribute) — Hint that a function is unlikely to be called.
  - [`no_builtins`](https://doc.rust-lang.org/reference/attributes/codegen.html#the-no_builtins-attribute) — Disables use of certain built-in functions.
  - [`target_feature`](https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute) — Configure platform-specific code generation.
  - [`track_caller`](https://doc.rust-lang.org/reference/attributes/codegen.html#the-track_caller-attribute) - Pass the parent call location to `std::panic::Location::caller()`.
  - [`instruction_set`](https://doc.rust-lang.org/reference/attributes/codegen.html#the-instruction_set-attribute) - Specify the instruction set used to generate a functions code
- Documentation
  - `doc` — Specifies documentation. See [The Rustdoc Book](https://doc.rust-lang.org/rustdoc/the-doc-attribute.html) for more information. [Doc comments](https://doc.rust-lang.org/reference/comments.html#doc-comments) are transformed into `doc` attributes.
- Preludes
  - [`no_std`](https://doc.rust-lang.org/reference/names/preludes.html#the-no_std-attribute) — Removes std from the prelude.
  - [`no_implicit_prelude`](https://doc.rust-lang.org/reference/names/preludes.html#the-no_implicit_prelude-attribute) — Disables prelude lookups within a module.
- Modules
  - [`path`](https://doc.rust-lang.org/reference/items/modules.html#the-path-attribute) — Specifies the filename for a module.
- Limits
  - [`recursion_limit`](https://doc.rust-lang.org/reference/attributes/limits.html#the-recursion_limit-attribute) — Sets the maximum recursion limit for certain compile-time operations.
  - [`type_length_limit`](https://doc.rust-lang.org/reference/attributes/limits.html#the-type_length_limit-attribute) — Sets the maximum size of a polymorphic type.
- Runtime
  - [`panic_handler`](https://doc.rust-lang.org/reference/runtime.html#the-panic_handler-attribute) — Sets the function to handle panics.
  - [`global_allocator`](https://doc.rust-lang.org/reference/runtime.html#the-global_allocator-attribute) — Sets the global memory allocator.
  - [`windows_subsystem`](https://doc.rust-lang.org/reference/runtime.html#the-windows_subsystem-attribute) — Specifies the windows subsystem to link with.
- Features
  - `feature` — Used to enable unstable or experimental compiler features. See [The Unstable Book](https://doc.rust-lang.org/unstable-book/index.html) for features implemented in `rustc`.
- Type System
  - [`non_exhaustive`](https://doc.rust-lang.org/reference/attributes/type_system.html#the-non_exhaustive-attribute) — Indicate that a type will have more fields/variants added in future.
- Debugger
  - [`debugger_visualizer`](https://doc.rust-lang.org/reference/attributes/debugger.html#the-debugger_visualizer-attribute) — Embeds a file that specifies debugger output for a type.
  - [`collapse_debuginfo`](https://doc.rust-lang.org/reference/attributes/debugger.html#the-collapse_debuginfo-attribute) — Controls how macro invocations are encoded in debuginfo.

### Derive 派生

- [derivable-traits](https://kaisery.github.io/trpl-zh-cn/appendix-03-derivable-traits.html)

*`derive`属性*允许为数据结构自动生成新的[程序项](https://rustwiki.org/zh-CN/reference/items.html)。它使用 [_MetaListPaths_](https://rustwiki.org/zh-CN/reference/attributes.html#meta-item-attribute-syntax)元项属性句法（为程序项）指定一系列要实现的 trait 或指定要执行的[派生宏](https://rustwiki.org/zh-CN/reference/procedural-macros.html#derive-macros)的路径。

例如，下面的派生属性将为结构体 `Foo` 创建一个实现 [`PartialEq`](https://doc.rust-lang.org/std/cmp/trait.PartialEq.html) trait 和 [`Clone`](https://doc.rust-lang.org/std/clone/trait.Clone.html) trait 的[实现(`impl` item)](https://rustwiki.org/zh-CN/reference/items/implementations.html)，类型参数 `T` 将被派生出的实现(`impl`)加上 `PartialEq` 或^[1](https://rustwiki.org/zh-CN/reference/attributes/derive.html#or-and)^ `Clone` 约束：

```rust
#[test]
#[should_panic(expected = "values don't match")]
fn mytest() {
    assert_eq!(1, 2, "values don't match");
}

```

1、自动派生常用的 traits： Rust 提供了一些常用的 traits，可以通过 `#[derive()]` 属性自动为结构体或枚举实现这些 traits。一些常见的可派生 traits 包括：

- `Debug`：通过实现 `Debug` trait，可以使用 `println!("{:?}", my_struct)` 来打印[结构体](https://zhida.zhihu.com/search?content_id=227981966&content_type=Article&match_order=3&q=%E7%BB%93%E6%9E%84%E4%BD%93&zhida_source=entity)的调试信息。
- `Clone`：通过实现 `Clone` trait，可以使用 `my_struct.clone()` 创建结构体的克隆副本。
- `PartialEq` 和 `Eq`：通过实现 `PartialEq` trait，可以进行结构体的部分相等性比较，而 `Eq` trait 则实现了完全相等性比较。
- `PartialOrd` 和 `Ord`：通过实现 `PartialOrd` trait，可以对结构体进行部分有序性比较，而 `Ord` trait 实现了完全有序性比较。

2、自定义 traits 的自动派生： 除了派生常见的 traits，您还可以自定义 traits，并使用 #[derive()] 属性为结构体或枚举自动生成实现代码。例如，如果您定义了一个名为 MyTrait 的自定义 trait，并希望为结构体自动实现它，可以这样写：

```rust
trait MyTrait {
    // trait 方法定义
}

#[derive(MyTrait)]
struct MyStruct {
    // 结构体字段
}
```

注意，对于自定义 traits 的派生，您需要在编写 MyTrait trait 时手动实现 #[derive(MyTrait)] 的逻辑，或者使用第三方库提供的派生宏。

3、手动实现 traits： 某些 traits 无法通过 #[derive()] 属性自动派生，因为它们可能需要更多的信息或自定义实现。在这种情况下，您需要手动为结构体或枚举实现这些 traits。手动实现 traits 通常涉及为每个 trait 方法提供具体的实现代码。以下是手动实现 MyTrait trait 的示例：

```rust
trait MyTrait {
    // trait 方法定义
}

struct MyStruct {
    // 结构体字段
}

impl MyTrait for MyStruct {
    // MyTrait 方法的具体实现
}
```

#### `#[derive[Debug]]`

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };

    println!("rect1 is {rect1:?}");
}
```

在 `{}` 中加入 `:?` 指示符告诉 `println!` 我们想要使用叫做 `Debug` 的输出格式。`Debug` 是一个 trait，它允许我们以一种对开发者有帮助的方式打印结构体，以便当我们调试代码时能看到它的值。

当我们有一个更大的结构体时，能有更易读一点的输出就好了，为此可以使用 `{:#?}` 替换 `println!` 字符串中的 `{:?}`。在这个例子中使用 `{:#?}` 风格将会输出如下：

```bash
$ cargo run
   Compiling rectangles v0.1.0 (file:///projects/rectangles)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.48s
     Running `target/debug/rectangles`
rect1 is Rectangle {
    width: 30,
    height: 50,
}

```

另一种使用 `Debug` 格式打印数值的方法是使用 [`dbg!` 宏](https://doc.rust-lang.org/std/macro.dbg.html)。`dbg!` 宏接收一个表达式的所有权（与 `println!` 宏相反，后者接收的是引用），打印出代码中调用 dbg! 宏时所在的文件和行号，以及该表达式的结果值，并返回该值的所有权。

注意：调用 `dbg!` 宏会打印到标准错误控制台流（`stderr`），与 `println!` 不同，后者会打印到标准输出控制台流（`stdout`）。

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

fn main() {
    let scale = 2;
    let rect1 = Rectangle {
        width: dbg!(30 * scale),
        height: 50,
    };

    dbg!(&rect1);
}
```

```bash
$ cargo run
   Compiling rectangles v0.1.0 (file:///projects/rectangles)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.61s
     Running `target/debug/rectangles`
[src/main.rs:10:16] 30 * scale = 60
[src/main.rs:14:5] &rect1 = Rectangle {
    width: 60,
    height: 50,
}
```

## Method and Impl

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };

    println!(
        "The area of the rectangle is {} square pixels.",
        rect1.area()
    );
}
```

为了使函数定义于 `Rectangle` 的上下文中，我们开始了一个 `impl` 块（`impl` 是 _implementation_ 的缩写），这个 `impl` 块中的所有内容都将与 `Rectangle` 类型相关联。接着将 `area` 函数移动到 `impl` 大括号中，并将签名中的第一个（在这里也是唯一一个）参数和函数体中其他地方的对应参数改成 `self`。然后在 `main` 中将我们先前调用 `area` 方法并传递 `rect1` 作为参数的地方，改成使用 **方法语法** （ _method syntax_ ）在 `Rectangle` 实例上调用 `area` 方法。方法语法获取一个实例并加上一个点号，后跟方法名、圆括号以及任何参数。

在 `area` 的签名中，使用 `&self` 来替代 `rectangle: &Rectangle`，`&self` 实际上是 `self: &Self` 的缩写。在一个 `impl` 块中，`Self` 类型是 `impl` 块的类型的别名。方法的第一个参数必须有一个名为 `self` 的 `Self` 类型的参数，所以 Rust 让你在第一个参数位置上只用 `self` 这个名字来简化。注意，我们仍然需要在 `self` 前面使用 `&` 来表示这个方法借用了 `Self` 实例，就像我们在 `rectangle: &Rectangle` 中做的那样。方法可以选择获得 `self` 的所有权，或者像我们这里一样不可变地借用 `self`，或者可变地借用 `self`，就跟其他参数一样。

这里选择 `&self` 的理由跟在函数版本中使用 `&Rectangle` 是相同的：我们并不想获取所有权，只希望能够读取结构体中的数据，而不是写入。如果想要在方法中改变调用方法的实例，需要将第一个参数改为 `&mut self`。通过仅仅使用 `self` 作为第一个参数来使方法获取实例的所有权是很少见的；这种技术通常用在当方法将 `self` 转换成别的实例的时候，这时我们想要防止调用者在转换之后使用原始的实例。

使用方法替代函数，除了可使用方法语法和不需要在每个函数签名中重复 `self` 的类型之外，其主要好处在于组织性。我们将某个类型实例能做的所有事情都一起放入 `impl` 块中，而不是让将来的用户在我们的库中到处寻找 `Rectangle` 的功能。

### 自动引用和解引用（ automatic referencing and dereferencing）

Rust 并没有一个与 `->` 等效的运算符；相反，Rust 有一个叫 **自动引用和解引用** （ _automatic referencing and dereferencing_ ）的功能。方法调用是 Rust 中少数几个拥有这种行为的地方。

它是这样工作的：当使用 `object.something()` 调用方法时，Rust 会自动为 `object` 添加 `&`、`&mut` 或 `*` 以便使 `object` 与方法签名匹配。

```rust
p1.distance(&p2);
(&p1).distance(&p2);
```

第一行看起来简洁的多。这种自动引用的行为之所以有效，是因为方法有一个明确的接收者———— `self` 的类型。在给出接收者和方法名的前提下，Rust 可以明确地计算出方法是仅仅读取（`&self`），做出修改（`&mut self`）或者是获取所有权（`self`）。事实上，Rust 对方法接收者的隐式借用让所有权在实践中更友好。

### [关联函数](https://kaisery.github.io/trpl-zh-cn/ch05-03-method-syntax.html#%E5%85%B3%E8%81%94%E5%87%BD%E6%95%B0)

所有在 `impl` 块中定义的函数被称为 **关联函数** （ _associated functions_ ），因为它们与 `impl` 后面命名的类型相关。我们可以定义不以 `self` 为第一参数的关联函数（因此不是方法），因为它们并不作用于一个结构体的实例。我们已经使用了一个这样的函数：在 `String` 类型上定义的 `String::from` 函数。

不是方法的关联函数经常被用作返回一个结构体新实例的构造函数。这些函数的名称通常为 `new` ，但 `new` 并不是一个关键字。例如我们可以提供一个叫做 `square` 关联函数，它接受一个维度参数并且同时作为宽和高，这样可以更轻松的创建一个正方形 `Rectangle` 而不必指定两次同样的值：

```rust
impl Rectangle {
    fn square(size: u32) -> Self {
        Self {
            width: size,
            height: size,
        }
    }
}
```

关键字 `Self` 在函数的返回类型中代指在 `impl` 关键字后出现的类型，在这里是 `Rectangle`

使用结构体名和 `::` 语法来调用这个关联函数：比如 `let sq = Rectangle::square(3);`。这个函数位于结构体的命名空间中：`::` 语法用于关联函数和模块创建的命名空间。[第七章](https://kaisery.github.io/trpl-zh-cn/ch07-02-defining-modules-to-control-scope-and-privacy.html)会讲到模块。

### [`Option` 枚举和其相对于空值的优势](https://kaisery.github.io/trpl-zh-cn/ch06-01-defining-an-enum.html#option-%E6%9E%9A%E4%B8%BE%E5%92%8C%E5%85%B6%E7%9B%B8%E5%AF%B9%E4%BA%8E%E7%A9%BA%E5%80%BC%E7%9A%84%E4%BC%98%E5%8A%BF)

这一部分会分析一个 `Option` 的案例，`Option` 是标准库定义的另一个枚举。`Option` 类型应用广泛因为它编码了一个非常普遍的场景，即一个值要么有值要么没值。

例如，如果请求一个非空列表的第一项，会得到一个值，如果请求一个空的列表，就什么也不会得到。从类型系统的角度来表达这个概念就意味着编译器需要检查是否处理了所有应该处理的情况，这样就可以避免在其他编程语言中非常常见的 bug。

编程语言的设计经常要考虑包含哪些功能，但考虑排除哪些功能也很重要。Rust 并没有很多其他语言中有的空值功能。 **空值** （_Null_ ）是一个值，它代表没有值。在有空值的语言中，变量总是这两种状态之一：空值和非空值。

Tony Hoare，null 的发明者，在他 2009 年的演讲 “Null References: The Billion Dollar Mistake” 中曾经说到：

```txt

I call it my billion-dollar mistake. At that time, I was designing the first comprehensive type system for references in an object-oriented language. My goal was to ensure that all use of references should be absolutely safe, with checking performed automatically by the compiler. But I couldn't resist the temptation to put in a null reference, simply because it was so easy to implement. This has led to innumerable errors, vulnerabilities, and system crashes, which have probably caused a billion dollars of pain and damage in the last forty years.

我称之为我十亿美元的错误。当时，我在为一个面向对象语言设计第一个综合性的面向引用的类型系统。我的目标是通过编译器的自动检查来保证所有引用的使用都应该是绝对安全的。不过我未能抵抗住引入一个空引用的诱惑，仅仅是因为它是这么的容易实现。这引发了无数错误、漏洞和系统崩溃，在之后的四十多年中造成了数十亿美元的苦痛和伤害。
```

空值的问题在于当你尝试像一个非空值那样使用一个空值，会出现某种形式的错误。因为空和非空的属性无处不在，非常容易出现这类错误。

然而，空值尝试表达的概念仍然是有意义的：空值是一个因为某种原因目前无效或缺失的值。

问题不在于概念而在于具体的实现。为此，Rust 并没有空值，不过它确实拥有一个可以编码存在或不存在概念的枚举。这个枚举是 `Option<T>`，而且它[定义于标准库中](https://doc.rust-lang.org/std/option/enum.Option.html)，如下：

```rust
enum Option<T> {
    None,
    Some(T),
}


```

`Option<T>` 枚举是如此有用以至于它甚至被包含在了 prelude 之中，你不需要将其显式引入作用域。另外，它的成员也是如此，可以不需要 `Option::` 前缀来直接使用 `Some` 和 `None`。即便如此 `Option<T>` 也仍是常规的枚举，`Some(T)` 和 `None` 仍是 `Option<T>` 的成员。

`<T>` 语法是一个我们还未讲到的 Rust 功能。它是一个泛型类型参数，第十章会更详细的讲解泛型。目前，所有你需要知道的就是 `<T>` 意味着 `Option` 枚举的 `Some` 成员可以包含任意类型的数据，同时每一个用于 `T` 位置的具体类型使得 `Option<T>` 整体作为不同的类型。这里是一些包含数字类型和字符串类型 `Option` 值的例子：

```rust
let some_number = Some(5);
let some_char = Some('e');

let absent_number: Option<i32> = None;

```

`some_number` 的类型是 `Option<i32>`。`some_char` 的类型是 `Option<char>`，是不同于 `some_number`的类型。因为我们在 `Some` 成员中指定了值，Rust 可以推断其类型。对于 `absent_number`，Rust 需要我们指定 `Option` 整体的类型，因为编译器只通过 `None` 值无法推断出 `Some` 成员保存的值的类型。这里我们告诉 Rust 希望 `absent_number` 是 `Option<i32>` 类型的。

当有一个 `Some` 值时，我们就知道存在一个值，而这个值保存在 `Some` 中。当有个 `None` 值时，在某种意义上，它跟空值具有相同的意义：并没有一个有效的值。那么，`Option<T>` 为什么就比空值要好呢？

简而言之，因为 `Option<T>` 和 `T`（这里 `T` 可以是任何类型）是不同的类型，编译器不允许像一个肯定有效的值那样使用 `Option<T>`。例如，这段代码不能编译，因为它尝试将 `Option<i8>` 与 `i8` 相加：

```rust
let x: i8 = 5;
let y: Option<i8> = Some(5);

let sum = x + y;
```

```bash
$ cargo run
   Compiling enums v0.1.0 (file:///projects/enums)
error[E0277]: cannot add `Option<i8>` to `i8`
 --> src/main.rs:5:17
  |
5 |     let sum = x + y;
  |                 ^ no implementation for `i8 + Option<i8>`
  |
  = help: the trait `Add<Option<i8>>` is not implemented for `i8`
  = help: the following other types implement trait `Add<Rhs>`:
            `&'a i8` implements `Add<i8>`
            `&i8` implements `Add<&i8>`
            `i8` implements `Add<&i8>`
            `i8` implements `Add`

For more information about this error, try `rustc --explain E0277`.
error: could not compile `enums` (bin "enums") due to 1 previous error
```

很好！事实上，错误信息意味着 Rust 不知道该如何将 `Option<i8>` 与 `i8` 相加，因为它们的类型不同。当在 Rust 中拥有一个像 `i8` 这样类型的值时，编译器确保它总是有一个有效的值。我们可以自信使用而无需做空值检查。只有当使用 `Option<i8>`（或者任何用到的类型）的时候需要担心可能没有值，而编译器会确保我们在使用值之前处理了为空的情况。

换句话说，在对 `Option<T>` 进行运算之前必须将其转换为 `T`。通常这能帮助我们捕获到空值最常见的问题之一：假设某值不为空但实际上为空的情况。

消除了错误地假设一个非空值的风险，会让你对代码更加有信心。为了拥有一个可能为空的值，你必须要显式的将其放入对应类型的 `Option<T>` 中。接着，当使用这个值时，必须明确的处理值为空的情况。只要一个值不是 `Option<T>` 类型，你就 **可以** 安全的认定它的值不为空。这是 Rust 的一个经过深思熟虑的设计决策，来限制空值的泛滥以增加 Rust 代码的安全性。

那么当有一个 `Option<T>` 的值时，如何从 `Some` 成员中取出 `T` 的值来使用它呢？`Option<T>` 枚举拥有大量用于各种情况的方法：你可以查看[它的文档](https://doc.rust-lang.org/std/option/enum.Option.html)。熟悉 `Option<T>` 的方法将对你的 Rust 之旅非常有用。

总的来说，为了使用 `Option<T>` 值，需要编写处理每个成员的代码。你想要一些代码只当拥有 `Some(T)` 值时运行，允许这些代码使用其中的 `T`。也希望一些代码只在值为 `None` 时运行，这些代码并没有一个可用的 `T` 值。`match` 表达式就是这么一个处理枚举的控制流结构：它会根据枚举的成员运行不同的代码，这些代码可以使用匹配到的值中的数据。

PS: C/C++如果程序员不够小心注意，很容易造成有空指针的情况，并且忘了检查，就会导致对空指针的使用！！（CSC-2024 C++ 惨痛的教训 - 依赖关系分析遍中没有对函数可能传入空指针进行检查，导致对空指针的操作！）

为什么要引入空指针？如果不得不引入某种为“空”的状态，最好 wrap 一下！！就像 rust 中的 Option 一样！
