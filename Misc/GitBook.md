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
