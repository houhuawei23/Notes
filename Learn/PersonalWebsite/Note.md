# Note

- [How to Create a Static Website with Jekyll](https://www.taniarascia.com/make-a-static-website-with-jekyll/)
- [Andrej Karpathy blog - 2014](https://karpathy.github.io/)
- [Why you (yes, you) should blog - 2017](https://medium.com/@racheltho/why-you-yes-you-should-blog-7d2544ac1045)
- [Hexo Theme Nexmoe](https://docs.nexmoe.com/)
- [pages.github](https://pages.github.com/)
- [hhw-google-blogger](https://www.blogger.com/blog/posts/7475310421209453726)

## Static Site Generator

A static site generator builds a website using plain HTML files. When a user visits a website created by a static site generator, it is loaded no differently than if you had created a website with plain HTML. By contrast, a dynamic site running on a server side language, such as PHP, must be built every time a user visits the site.

静态站点生成器使用纯 HTML 文件构建网站。当用户访问由静态网站生成器创建的网站时，其加载方式与您使用纯 HTML 创建网站时没有什么不同。相比之下，在服务器端语言（例如 PHP）上运行的动态站点必须在用户每次访问该站点时构建。

You can treat a static site generator as a very simple sort of CMS (content management system). Instead of having to include your entire header and footer on every page, for example, you can create a header.html and footer.html and load them into each page. Instead of having to write in HTML, you can write in Markdown, which is much faster and more efficient.

您可以将静态站点生成器视为一种非常简单的 CMS（内容管理系统）。例如，您不必在每个页面上包含整个页眉和页脚，您可以创建 header.html 和 footer.html 并将它们加载到每个页面中。您不必使用 HTML 编写，而是可以使用 Markdown 编写，这样更快、更高效。

Here are some of the main advantages of static site generators over dynamic sites:  
以下是静态站点生成器相对于动态站点的一些主要优点：

- **Speed**: your website will perform much faster, as the server does not need to parse any content. It only needs to read plain HTML.
  - **速度**: 您的网站将执行得更快，因为服务器不需要解析任何内容。它只需要读取纯 HTML。
- **Security**: your website will be much less vulnerable to attacks, since there is nothing that can be exploited server side.
  - **安全性**: 您的网站将更不容易受到攻击，因为服务器端没有任何东西可以被利用。
- **Simplicity**: there are no databases or programming languages to deal with. A simple knowledge of HTML and CSS is enough.
  - **简单性**: 无需处理数据库或编程语言。了解 HTML 和 CSS 的简单知识就足够了。
- **Flexibility**: you know exactly how your site works, as you made it from scratch.
  - **灵活性**: 您确切地知道您的网站是如何运作的，因为您是从头开始创建的。

Of course, dynamic sites have their advantages as well. The addition of an admin panel makes for ease of updating, especially for those who are not tech-savvy. Generally, a static site generator would not be the best idea for making a CMS for a client. Static site generators also don't have the possibility of updating with real time content. It's important to understand how both work to know what would work best for your particular project.

当然，动态网站也有其优点。添加管理面板可以轻松更新，特别是对于那些不精通技术的人来说。一般来说，静态站点生成器并不是为客户制作 CMS 的最佳主意。静态站点生成器也无法更新实时内容。了解两者的工作原理非常重要，这样才能知道哪种方法最适合您的特定项目。

## Self Host Docs Wibesite

- gitbook: origin project no more maintain, turn into Gitbook.
  - old still have bugs
- [mdBook - rust](https://github.com/rust-lang/mdBook)
  - math delimiters support sick
  - [text](https://hellowac.github.io/mdbook-doc-zh/index.html)
- [retype](https://retype.com/)
  - free plan only support 100 pages
- [mkdocs](https://www.mkdocs.org/)
- [mkdocs-material](https://squidfunk.github.io/mkdocs-material/)
  - math delimiters support great, 'dollars' and 'brackets'
  - 'dollars': `$xxx$` and `$$xxx$$`
  - 'brackets': `\(...\)` and `\[...\]`
- [docusaurus](https://docusaurus.io/)
  - init project stack, dont know why
- [bookstackapp](https://www.bookstackapp.com/)
  - A platform to create documentation/wiki content built with PHP & Laravel
- [Archbee](https://www.archbee.com/)
  - like notion...
- [hyperbook](https://hyperbook.openpatch.org/)
  - Hyperbook is a quick and easy way to build interactive workbooks, that support modern standards and runs superfast.
  - looks like great!
  - cant release port?? may have bugs, too small depelop group.
