# GRUB

GRUB (GRand Unified Bootloader) is the default bootloader for most Linux distributions. It is responsible for loading the kernel and passing control to it. It can be customized to load different operating systems, different kernel versions, and even different boot configurations.

To customize GRUB, you can edit the `/etc/default/grub` file. This file contains a list of configuration options for GRUB. You can add or remove options as needed.

To make changes permanent, you can update the GRUB configuration using the `update-grub` command. This will generate a new `/boot/grub/grub.cfg` file that will be used by GRUB on the next boot.

To see the current GRUB configuration, you can run the `grub-editenv` command. This will open an interactive shell where you can modify the GRUB environment variables.

To add a new operating system to GRUB, you can create a new entry in the `/etc/grub.d` directory. This directory contains a series of shell scripts that are executed by GRUB in order to generate the GRUB configuration. Each script is responsible for adding a new entry to the GRUB menu.

#### **GR**and **U**nified **B**ootloader（大一统引导程序）

GRUB 试图为 IBM PC 兼容机提供一个引导加载程序，它既能为初学者或对技术不感兴趣的用户提供方便，又能灵活地帮助专家在不同的环境中使用。目前，它最适用于至少使用一种类似 UNIX 的免费操作系统的用户，但也可用于大多数 PC 操作系统。

这个项目的起因实际上是我们想在 IBM PC 兼容系统的 Mach4 上以符合多重引导标准的方式引导 GNU HURD 操作系统。然后，我尝试在 FreeBSD 使用的标准引导加载程序中添加对额外功能的支持。为了让所有功能都能正常工作，我必须做的事情越来越多，直到显然有必要从头开始另起炉灶。

GRUB 从一开始的多模块引导加载器发展到现在已经有很长的路要走了。它所使用的一些技术在自由软件世界中是独一无二的，还有一些技术显然也优于大多数专有操作系统。这里和多引导建议中的文档对未来的操作系统和 PC 引导加载程序编写者应该非常有用。

grub 命令

grub 命令的功能是用于交互式地管理 GRUB 引导程序。GRUB 是一个系统引导程序，可以服务于 Linux、Windows、FreeBSD 等常见操作系统，配置方式分为交互式和非交互式两种模式，用户只需要键入 grub 命令即可进入到“grub>”提示状态，然后通过常用命令及参数进行配置工作。原文链接：

[https://www.linuxcool.com/grub](https://www.linuxcool.com/grub)

[Linux 黑话解释：Linux 中的 GRUB 是什么？ | Linux 中国 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/492509251)

[GNU GRUB - GNU Project - Free Software Foundation (FSF)](https://www.gnu.org/software/grub/)
