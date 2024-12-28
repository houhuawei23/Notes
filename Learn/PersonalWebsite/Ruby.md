# [Ruby](https://www.ruby-lang.org/en/)

- Ruby: A dynamic, open source programming language.
- RubyGems: A package manager for the Ruby programming language.
- Bundler: A tool for managing Ruby dependencies.

## Ruby

A dynamic, open source programming language with a focus on simplicity and productivity. It has an elegant syntax that is natural to read and easy to write.

一种动态的开源编程语言，重点关注 简单性和生产力。它有一个优雅的语法： 读起来自然，写起来也容易。

Features:

- Simple syntax,
- Basic OO features (classes, methods, objects, and so on),
- Special OO features (mixins, singleton methods, renaming, and so on),
- Operator overloading,
- Exception handling,
- Iterators and closures,
- Garbage collection,
- Dynamic loading (depending on the architecture),
- High transportability (runs on various Unices, Windows, DOS, macOS, OS/2, Amiga, and so on).

wikipedia:

**Ruby** is an [interpreted](https://en.wikipedia.org/wiki/Interpreted_language), [high-level](https://en.wikipedia.org/wiki/High-level_programming_language), [general-purpose programming language](https://en.wikipedia.org/wiki/General-purpose_programming_language). It was designed with an emphasis on programming productivity and simplicity. In Ruby, everything is an [object](https://en.wikipedia.org/wiki/Object_(computer_science)), including [primitive data types](https://en.wikipedia.org/wiki/Primitive_data_type). It was developed in the mid-1990s by [Yukihiro "Matz" Matsumoto](https://en.wikipedia.org/wiki/Yukihiro_Matsumoto) in [Japan](https://en.wikipedia.org/wiki/Japan).

**Ruby**是一种[解释型](https://en.wikipedia.org/wiki/Interpreted_language)、[高级](https://en.wikipedia.org/wiki/High-level_programming_language)、[通用编程语言](https://en.wikipedia.org/wiki/General-purpose_programming_language)。它的设计重点是编程效率和简单性。在 Ruby 中，一切都是[对象](https://en.wikipedia.org/wiki/Object_(computer_science))，包括[原始数据类型](https://en.wikipedia.org/wiki/Primitive_data_type)。它是由[日本](https://en.wikipedia.org/wiki/Japan)[松本幸弘“Matz”](https://en.wikipedia.org/wiki/Yukihiro_Matsumoto)于 20 世纪 90 年代中期开发的。

Ruby is [dynamically typed](https://en.wikipedia.org/wiki/Dynamic_typing) and uses [garbage collection](https://en.wikipedia.org/wiki/Garbage_collection_(computer_science)) and [just-in-time compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation). It supports multiple programming paradigms, including [procedural](https://en.wikipedia.org/wiki/Procedural_programming), [object-oriented](https://en.wikipedia.org/wiki/Object-oriented_programming), and [functional programming](https://en.wikipedia.org/wiki/Functional_programming). According to the creator, Ruby was influenced by [Perl](https://en.wikipedia.org/wiki/Perl), [Smalltalk](https://en.wikipedia.org/wiki/Smalltalk), [Eiffel](https://en.wikipedia.org/wiki/Eiffel_(programming_language)), [Ada](https://en.wikipedia.org/wiki/Ada_(programming_language)), [BASIC](https://en.wikipedia.org/wiki/BASIC), [Java](https://en.wikipedia.org/wiki/Java_(programming_language)), and [Lisp](https://en.wikipedia.org/wiki/Lisp_(programming_language)).[[10]](https://en.wikipedia.org/wiki/Ruby_(programming_language)#cite_note-about-10)[[3]](https://en.wikipedia.org/wiki/Ruby_(programming_language)#cite_note-confreaks-3)

Ruby 是[动态类型的](https://en.wikipedia.org/wiki/Dynamic_typing)，并使用[垃圾收集](https://en.wikipedia.org/wiki/Garbage_collection_(computer_science))和[即时编译](https://en.wikipedia.org/wiki/Just-in-time_compilation)。它支持多种编程范例，包括[过程式编程](https://en.wikipedia.org/wiki/Procedural_programming)、[面向对象编程](https://en.wikipedia.org/wiki/Object-oriented_programming)和[函数式编程](https://en.wikipedia.org/wiki/Functional_programming)。根据创建者的说法，Ruby 受到[Perl](https://en.wikipedia.org/wiki/Perl) 、 [Smalltalk](https://en.wikipedia.org/wiki/Smalltalk) 、 [Eiffel](https://en.wikipedia.org/wiki/Eiffel_(programming_language)) 、 [Ada](https://en.wikipedia.org/wiki/Ada_(programming_language)) 、 [BASIC](https://en.wikipedia.org/wiki/BASIC) 、 [Java](https://en.wikipedia.org/wiki/Java_(programming_language))和[Lisp](https://en.wikipedia.org/wiki/Lisp_(programming_language))的影响。 [[ 10 ]](https://en.wikipedia.org/wiki/Ruby_(programming_language)#cite_note-about-10) [[ 3 ]](https://en.wikipedia.org/wiki/Ruby_(programming_language)#cite_note-confreaks-3)

## [RubyGems](https://rubygems.org/)

- [wikipedia](https://en.wikipedia.org/wiki/RubyGems)

RubyGems is a package manager for the Ruby programming language that provides a standard format for distributing Ruby programs and libraries (in a self-contained format called a "gem"), a tool designed to easily manage the installation of gems, and a server for distributing them. It was created by Chad Fowler, Jim Weirich, David Alan Black, Paul Brannan and Richard Kilmer in 2004.[2]

RubyGems 是 Ruby 编程语言的包管理器，它提供了用于分发 Ruby 程序和库的标准格式（以称为“gem”的独立格式）、一个旨在轻松管理 gems 安装的工具以及一个用于分发它们。它由 Chad Fowler 、 Jim Weirich 、 David Alan Black 、 Paul Brannan 和 Richard Kilmer 于 2004 年创建。 [ 2 ].

The interface for RubyGems is a command-line tool called gem which can install and manage libraries (the gems).[3] RubyGems integrates with Ruby run-time loader to help find and load installed gems from standardized library folders. Though it is possible to use a private RubyGems repository, the public repository is most commonly used for gem management.

RubyGems 的界面是一个名为 gem 的命令行工具，它可以安装和管理库（gems）。 [ 3 ] RubyGems 与 Ruby 运行时加载器集成，以帮助从标准化库文件夹中查找并加载已安装的 gem。尽管可以使用私有 RubyGems 存储库，但公共存储库最常用于 gem 管理。

The public repository helps users find gems, resolve dependencies and install them. RubyGems is bundled with the standard Ruby package as of Ruby 1.9.[4]

公共存储库可帮助用户查找 gem、解决依赖关系并安装它们。从 Ruby 1.9 开始，RubyGems 与标准 Ruby 包捆绑在一起。 [ 4 ]

## [Bundler](https://bundler.io/)

Bundler provides a consistent environment for Ruby projects by tracking and installing the exact gems and versions that are needed.

Bundler 通过跟踪和安装所需的确切 gem 和版本，为 Ruby 项目提供一致的环境。

Bundler is an exit from dependency hell, and ensures that the gems you need are present in development, staging, and production. Starting work on a project is as simple as bundle install.

Bundler 是摆脱依赖地狱的出口，并确保您需要的 gem 存在于开发、登台和生产中。开始项目工作就像 bundle install 一样简单。

## Ruby-101 in jekyll

Gems

- **Gems** are code you can include in Ruby projects. Gems package specific functionality. You can share gems across multiple projects or with other people. Gems can perform actions like:
- Gems 是可以包含在 Ruby 项目中的代码。 Gems 封装特定功能。您可以跨多个项目或与其他人共享 gem。

Gemfile

- A **Gemfile** is a list of gems used by your site. Every Jekyll site has a Gemfile in the main folder.
- Gemfile 是您的站点使用的 gem 列表。每个 Jekyll 站点的主文件夹中都有一个 Gemfile。

Bundler

- **Bundler** is a gem that installs all gems in your Gemfile.
- To install gems in your Gemfile using Bundler, run the following in the directory that has the Gemfile:

```bash
bundle install
bundle exec jekyll serve
```

- To bypass Bundler if you aren’t using a Gemfile, run `jekyll serve`.

## Ruby Version Manager ([RVM](https://rvm.io/))