# GitBook

Product documentation (your users will love)

Forget building your own custom docs platform. With GitBook you get beautiful documentation for your users, and a branch-based Git workflow for your team.

- [gitbook.com](https://www.gitbook.com/)
- [gitbook-documentation zh](https://chrisniael.gitbooks.io/gitbook-documentation/content/index.html)
- [gitbook-cli](https://github.com/GitbookIO/gitbook-cli)

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
