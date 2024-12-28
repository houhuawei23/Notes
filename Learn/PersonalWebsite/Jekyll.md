## [Jekyll](https://jekyllrb.com/)

> Transform your plain text into static websites and blogs.\
> 将纯文本转换为静态网站和博客。

Jekyll is a **static site generator** that runs on the Ruby programming language.

- Simple 简单的
  - No more databases, comment moderation, or pesky updates to install—just your content.
  - 不再需要安装数据库、评论审核或烦人的更新，只需安装您的内容。
  - [How Jekyll works ](https://jekyllrb.com/docs/usage/)
- Static 静止的
  - [Markdown](https://daringfireball.net/projects/markdown/), [Liquid](https://github.com/Shopify/liquid/wiki), HTML & CSS go in. Static sites come out ready for deployment.
  - 加入[Markdown](https://daringfireball.net/projects/markdown/) 、 [Liquid](https://github.com/Shopify/liquid/wiki) 、HTML 和 CSS。静态站点出来后即可部署。
  - [Jekyll template guide ](https://jekyllrb.com/docs/templates/)
- Blog-aware 博客意识
  - Permalinks, categories, pages, posts, and custom layouts are all first-class citizens here.
  - 永久链接、类别、页面、帖子和自定义布局在这里都是一等公民。
  - [Migrate your blog ](https://import.jekyllrb.com/)

Running in seconds, [Quickstart](https://jekyllrb.com/docs/):

```bash
# make sure satisfy the [prerequisites](https://jekyllrb.com/docs/installation/#requirements)
# on Debian:
sudo apt-get install ruby-full build-essential
# on Ubuntu:
sudo apt-get install ruby-full build-essential zlib1g-dev

# Install the jekyll and bundler gems
gem install bundler jekyll
jekyll new my-awesome-site
cd my-awesome-site
bundle exec jekyll serve # --livereload
# => Now browse to http://localhost:4000
```

Problems:

```bash
gem install bundler jekyll
# Error:
# /usr/bin/ruby3.1 -I /usr/lib/ruby/vendor_ruby -r ./siteconf20241228-103530-c2f02s.rb extconf.rb
# mkmf.rb can't find header files for ruby at /usr/lib/ruby/include/ruby.h

# Solution:
sudo apt install ruby-dev

# Error
# sudo gem install jekyll
# Net::OpenTimeout: Failed to open TCP connection to github.com:443 (Connection timed out - user specified timeout)

# Solution:
# network error, just use proxy.
```
