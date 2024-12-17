# PEG: Parsing Expression Grammar 解析表达式语法

- [Parsing_expression_grammar](https://en.wikipedia.org/wiki/Parsing_expression_grammar)
- [awesome-pest](https://github.com/pest-parser/awesome-pest)

In computer science, a parsing expression grammar (PEG) is a type of analytic formal grammar, i.e. it describes a formal language in terms of a set of rules for recognizing strings in the language. The formalism was introduced by Bryan Ford in 2004[1] and is closely related to the family of top-down parsing languages introduced in the early 1970s. Syntactically, PEGs also look similar to context-free grammars (CFGs), but they have a different interpretation: the choice operator selects the first match in PEG, while it is ambiguous in CFG. This is closer to how string recognition tends to be done in practice, e.g. by a recursive descent parser.

在计算机科学中，解析表达式语法（ PEG ）是一种分析形式语法，即它根据一组用于识别语言中字符串的规则来描述形式语言。该形式主义由 Bryan Ford 于 20041 年提出，与 20 世纪 70 年代初推出的自顶向下解析语言系列密切相关。从语法上讲，PEG 看起来也与上下文无关语法（CFG）类似，但它们有不同的解释：选择运算符选择 PEG 中的第一个匹配项，而在 CFG 中它是不明确的。这更接近于实践中字符串识别的方式，例如通过递归下降解析器。

Unlike CFGs, PEGs cannot be ambiguous; a string has exactly one valid parse tree or none. It is conjectured that there exist context-free languages that cannot be recognized by a PEG, but this is not yet proven.[1] PEGs are well-suited to parsing computer languages (and artificial human languages such as Lojban) where multiple interpretation alternatives can be disambiguated locally, but are less likely to be useful for parsing natural languages where disambiguation may have to be global.[2]

与 CFG 不同，PEG 不能含糊不清；一个字符串只有一个有效的解析树，或者没有。据推测，存在 PEG 无法识别的上下文无关语言，但这尚未得到证实。1 PEG 非常适合解析计算机语言（以及Lojban等人工人类语言），其中可以有多种解释替代方案。在本地消除歧义，但对于解析可能必须是全局消歧的自然语言不太有用。
