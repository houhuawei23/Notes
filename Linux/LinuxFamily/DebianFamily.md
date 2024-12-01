## Debian12 bookworm Release Notes

5.1.5. Fcitx versions no longer co-installable
5.1.5. Fcitx 版本不再可共同安装

The packages fcitx and fcitx5 provide version 4 and version 5 of the popular Fcitx Input Method Framework. Following upstream's recommendation, they can no longer be co-installed on the same operating system. Users should determine which version of Fcitx is to be kept if they had co-installed fcitx and fcitx5 previously.
软件包 fcitx 和 fcitx5 提供流行的 Fcitx 输入法框架的版本 4 和版本 5。根据上游的建议，它们不能再共同安装在同一个操作系统上。如果用户之前已经共同安装了 fcitx 和 fcitx5，则应确定要保留哪个版本的 fcitx。

Before the upgrade, users are strongly encouraged to purge all related packages for the unwanted Fcitx version (fcitx-* for Fcitx 4, and fcitx5-* for Fcitx 5). When the upgrade is finished, consider executing the im-config again to select the desired input method framework to be used in the system.
在升级之前，强烈建议用户清除不需要的 Fcitx 版本的所有相关软件包（Fcitx 4 的 fcitx-* 和 Fcitx 5 的 fcitx5-*）。升级完成后，考虑再次执行 im-config 选择要在系统中使用的所需输入法框架。