# Typst

## Links

- [Typst](https://typst.io/)
- [polylux: package to create presentation slides.](https://polylux.dev/book/polylux.html)

## Reference

### `#show` rule

```typst
#show "ArtosFlow": name => box[
  #box(image(
    "logo.svg",
    height: 0.7em,
  ))
  #name
]

This report is embedded in the
ArtosFlow project. ArtosFlow is a
project of the Artos Institute.
```

With show rules, you can redefine how Typst displays certain elements. You specify which elements Typst should show differently and how they should look. Show rules can be applied to instances of text, many functions, and even the whole document.
使用 show rules，您可以重新定义 Typst 显示某些元素的方式。您可以指定 Typst 应该以不同的方式显示哪些元素以及它们应该如何显示。显示规则可以应用于文本实例、许多函数，甚至整个文档。

There is a lot of new syntax in this example: We write the show keyword, followed by a string of text we want to show differently and a colon. Then, we write a function that takes the content that shall be shown as an argument. Here, we called that argument name. We can now use the name variable in the function's body to print the ArtosFlow name. Our show rule adds the logo image in front of the name and puts the result into a box to prevent linebreaks from occurring between logo and name. The image is also put inside of a box, so that it does not appear in its own paragraph.

这个例子中有很多新的语法：我们编写 show 关键字，后跟我们想要以不同方式显示的一串文本和一个冒号。然后，我们编写一个函数，该函数将应显示为参数的内容。在这里，我们将该参数称为 name。我们现在可以使用函数体中的 name 变量来打印 ArtosFlow 名称。我们的 show 规则将 logo 图像添加到名称前面，并将结果放入一个框中，以防止 logo 和 name 之间出现换行。图像也被放在一个框内，这样它就不会出现在自己的段落中。

