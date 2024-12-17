# VScode

## Shortcuts

- `Ctrl+p` 快速打开文件
- `Ctrl+shift+p` 打开命令面板
- `Ctrl+w` 关闭当前文件
- `Ctrl+k+w` 关闭所有文件
- `Ctrl+f` 快速搜索
- `Ctrl+g` 跳转到指定行
- `Ctrl+Alt+f` Find in Explorer
- `Ctrl+R` Run Recent Command

## Others

#### 文件恢复

在 vscode 界面，按住 `Ctrl+shift+p` 打开命令面板，找到本地历史记录

#### VScode server

确认 server commit_id

```Bash
# ~/.vscode-server/bin
# 下载对应的server程序
# （注意把:${commit_id}替换成对应的Commit ID）
https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable
# vscode-server-linux-x64.tar.gz
# 放到 ~/.vscode-server/bin/${commit_id}/ 文件夹下
# 解压
tar -zxvf vscode-server-linux-x64.tar.gz --strip=1

https://update.code.visualstudio.com/commit:e7e037083ff4455cf320e344325dacb480062c3c/server-linux-x64/stable

```
