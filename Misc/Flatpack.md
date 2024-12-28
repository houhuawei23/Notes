# Flatpack

- [flatpak](https://flatpak.org/)

The future of apps on Linux.

Change Sources:

```bash
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo

flatpak remotes --show-details  # 显示flatpak官方源

flatpak remote-modify flathub --url=https://mirror.sjtu.edu.cn/flathub

# run gimp
flatpak run org.gimp.GIMP//stable
```