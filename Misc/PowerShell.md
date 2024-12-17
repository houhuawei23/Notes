# PowerShell

## 常用命令

### Proxy settings

```PowerShell
# C:\Users\ludiser\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1

function set_proxy0 { $env:HTTP_PROXY="http://127.0.0.1:7890" }
function set_proxy1 { $env:HTTPS_PROXY="https://127.0.0.1:7890" }
function unset_proxy0 { $env:HTTP_PROXY="" }
function unset_proxy1 { $env:HTTPS_PROXY="" }
```

### 删除

```PowerShell
# 删除指定文件
Remove-Item * -Include *.json -Recurse
# 删除文件而保留文件夹
# 「This example deletes all of the files that have names that include a dot (.) 」
Remove-Item * -Include *.* -Exclude *.md -Recurse
# 删除包含指定字符的文件夹
# 一定要注意加上通配符「*bin*」，否则只会删除bin这样的文件夹
Remove-Item * -Recurse -Include *bin*

```

- 获取版本信息 `$PSVersionTable.PSVersion`

- 获取主机信息 `Get-Host`

- 下载文件 `Invoke-WebRequest` / `iwr`
- 命令 `get-command`
- 进程 `get-process`
- 指令重命名 `Set-Alias xxx0 xxx1`

- 清屏`cls`
- 查找程序路径 `Get-Command -Name xxname`
