# GitBook

Product documentation (your users will love)

Forget building your own custom docs platform. With GitBook you get beautiful documentation for your users, and a branch-based Git workflow for your team.

- [gitbook.com](https://www.gitbook.com/)
- [gitbook-ng.github.io](https://gitbook-ng.github.io/)
- [gitbook-documentation zh](https://chrisniael.gitbooks.io/gitbook-documentation/content/index.html)
- [gitbook-cli](https://github.com/GitbookIO/gitbook-cli)
- [github: GitbookIO/gitbook](https://github.com/GitbookIO/gitbook)
- [GitbookIO/integrations](https://github.com/GitbookIO/integrations)
- [Gitbook 打造的 Gitbook 说明文档](https://www.mapull.com/gitbook/comscore/)

GitBook 是基于 Node.js 的开源命令行工具，用于输出漂亮的电子书。

遗憾的是，GitBook开源项目已经停止维护，专注打造的 gitbook.com 网站在国内访问受限。

## Extensions

- [awesome-gitbook-plugins](https://github.com/swapagarwal/awesome-gitbook-plugins?tab=readme-ov-file)
- [include-codeblock](https://github.com/azu/gitbook-plugin-include-codeblock)
- [edit-link](https://github.com/rtCamp/gitbook-plugin-edit-link)
- [sharing](https://github.com/GitbookIO/plugin-sharing)
- [terminull](https://github.com/ridaeh/gitbook-plugin-terminull)
- [intopic-toc](https://github.com/fzankl/gitbook-plugin-intopic-toc)
- [disqus](https://github.com/GitbookIO/plugin-disqus)
- [github](https://github.com/GitbookIO/plugin-github)
- [back-to-top-button](https://github.com/stuebersystems/gitbook-plugin-back-to-top-button)
- [download-pdf-link](https://github.com/show0k/gitbook-plugin-download-pdf-link)
- [mermaid-newface](https://github.com/TakuroFukamizu/gitbook-plugin-mermaid-newface)

## Install

```bash
# install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# install node version 10.24.1
nvm install 10.24.1

# install gitbook-cli (with change npm source)
npm config set registry https://registry.npmmirror.com
npm install gitbook-cli -g

# install gitbook
gitbook -V

# 下载添加的插件 & build
gitbook install
gitbook build # generate static files under `_book` directory

# start server: localhost:4000
gitbook serve
```

## 格式

格式主要注重简单和易读性

GitBook 约定了下面这些文件的作用：

- README：书本的介绍
- SUMMARY：章节结构, 用来生成书本内容的预览表。
- LANGS：多语言书籍
- GLOSSARY：术语描述的清单

至少需要一个 README 和 SUMMARY 文件来构建一本书。

## Gitbook pdf

```bash
gitbook pdf <gitbook-folder-location> <pdf-location>.pdf
```