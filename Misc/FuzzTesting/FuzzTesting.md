# Fuzz Testing

[wiki: Fuzz testing](https://en.wikipedia.org/wiki/Fuzzing)

In [programming](https://en.wikipedia.org/wiki/Computer_programming "Computer programming") and [software development](https://en.wikipedia.org/wiki/Software_development "Software development"), **fuzzing** or **fuzz testing** is an automated [software testing](https://en.wikipedia.org/wiki/Software_testing "Software testing") technique that involves providing invalid, unexpected, or [random data](https://en.wikipedia.org/wiki/Random_data "Random data") as inputs to a [computer program](https://en.wikipedia.org/wiki/Computer_program "Computer program"). The program is then monitored for exceptions such as [crashes](https://en.wikipedia.org/wiki/Crash_(computing)) "Crash (computing)"), failing built-in code [assertions](https://en.wikipedia.org/wiki/Assertion_(software_development)) "Assertion (software development)"), or potential [memory leaks](https://en.wikipedia.org/wiki/Memory_leak "Memory leak"). Typically, fuzzers are used to test programs that take structured inputs. This structure is specified, such as in a [file format](https://en.wikipedia.org/wiki/File_format "File format") or [protocol](https://en.wikipedia.org/wiki/Communications_protocol "Communications protocol") and distinguishes valid from invalid input. An effective fuzzer generates semi-valid inputs that are "valid enough" in that they are not directly rejected by the parser, but do create unexpected behaviors deeper in the program and are "invalid enough" to expose [corner cases](https://en.wikipedia.org/wiki/Corner_case "Corner case") that have not been properly dealt with.

在编程和软件开发中，模糊测试或模糊测试是一种自动化软件测试技术，涉及提供无效、意外或随机的数据作为计算机程序的输入。然后，将监视程序是否存在异常，例如崩溃、内置代码断言失败或潜在的内存泄漏。通常，模糊测试程序用于测试采用结构化输入的程序。此结构是指定的，例如以文件格式或协议指定，并区分有效输入和无效输入。有效的模糊测试器会生成“足够有效”的半有效输入，因为它们不会被解析器直接拒绝，但确实会在程序的更深处产生意想不到的行为，并且“足够无效”以暴露未正确处理的极端情况。

- [afl](https://lcamtuf.coredump.cx/afl/)

## AFL: American Fuzzy Lop

- [afl](https://lcamtuf.coredump.cx/afl/)
- [aflplusplus](https://aflplus.plus/)
- [afl.rs](https://github.com/rust-fuzz/afl.rs)
- [rust-fuzz: AFL for Rust](https://rust-fuzz.github.io/book/afl.html)

*American fuzzy lop* is a security-oriented [fuzzer](https://en.wikipedia.org/wiki/Fuzz_testing) that employs a novel type of compile-time instrumentation and genetic algorithms to automatically discover clean, interesting test cases that trigger new internal states in the targeted binary. This substantially improves the functional coverage for the fuzzed code. The compact [synthesized corpora](https://lcamtuf.coredump.cx/afl/demo/) produced by the tool are also useful for seeding other, more labor- or resource-intensive testing regimes down the road.

American fuzzy lop 是一个以安全为导向的 模糊器，采用新型编译时检测和遗传算法来自动发现干净、有趣的测试用例，从而触发目标二进制文件中的新内部状态。这大大提高了模糊代码的功能覆盖率。该工具生成的紧凑的合成语料库也可用于为将来的其他劳动或资源密集型测试制度奠定基础。

Compared to other instrumented fuzzers, *afl-fuzz* is designed to be practical: it has modest performance overhead, uses a variety of highly effective fuzzing strategies and effort minimization tricks, requires [essentially no configuration](https://lcamtuf.coredump.cx/afl/QuickStartGuide.txt), and seamlessly handles complex, real-world use cases - say, common image parsing or file compression libraries.

与其他仪器化模糊器相比,afl-fuzz被设计为实用:它具有适度的性能开销,使用各种高效的模糊策略和努力最小化技巧,基本上不需要配置,并且可以无缝处理复杂的现实世界用例 - 例如,常见的图像解析或文件压缩库。

### AFL.RS

[Fuzz testing](https://en.wikipedia.org/wiki/Fuzz_testing) is a software testing technique used to find security and stability issues by providing pseudo-random data as input to the software. [AFLplusplus](https://aflplus.plus/) is a popular, effective, and modern fuzz testing tool based on [AFL](http://lcamtuf.coredump.cx/afl/). This library, afl.rs, allows one to run AFLplusplus on code written in [the Rust programming language](https://www.rust-lang.org/).

模糊测试是一种软件测试技术，用于通过提供伪随机数据作为软件的输入来发现安全性和稳定性问题。 AFLplusplus 是一种基于 AFL 的流行、有效、现代的模糊测试工具。这个库 afl.rs 允许人们在用 Rust 编程语言编写的代码上运行 AFLplusplus。
