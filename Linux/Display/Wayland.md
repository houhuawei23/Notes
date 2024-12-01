

# Wayland
[text](https://wayland.freedesktop.org/)
[text](https://wiki.archlinux.org/title/Wayland)

Wayland is a replacement for the X11 window system protocol and architecture with the aim to be easier to develop, extend, and maintain.
Wayland 是 X11 窗口系统协议和架构的替代品，旨在更易于开发、扩展和维护。

Wayland is the language (protocol) that applications can use to talk to a display server in order to make themselves visible and get input from the user (a person). A Wayland server is called a "compositor". Applications are Wayland clients.
Wayland 是应用程序可以用来与显示服务器通信的语言（协议），以使自身可见并从用户（一个人）那里获得输入。Wayland 服务器称为“合成器”。应用程序是 Wayland 客户端。

Wayland also refers to a system architecture. It is not just a server-client relationship between a compositor and applications. There is no single common Wayland server like Xorg is for X11, but every graphical environment brings with it one of many compositor implementations. Window management and the end user experience are often tied to the compositor rather than swappable components.
Wayland 还指系统架构。它不仅仅是合成器和应用程序之间的服务器-客户端关系。没有像 Xorg 那样用于 X11 的单一通用 Wayland 服务器，但每个图形环境都带来了许多合成器实现之一。窗口管理和最终用户体验通常与合成器而不是可交换组件相关联。

A core part of Wayland architecture is libwayland: an inter-process communication library that translates a protocol definition in XML to a C language API. This library does not implement Wayland, it merely encodes and decodes Wayland messages. The actual implementations are in the various compositor and application toolkit projects.
Wayland 架构的核心部分是 libwayland：一个进程间通信库，可将 XML 中的协议定义转换为 C 语言 API。该库不实现 Wayland，它只是对 Wayland 消息进行编码和解码。实际实现位于各种合成器和应用程序工具包项目中。

Wayland does not restrict where and how it is used. A Wayland compositor could be a standalone display server running on Linux kernel modesetting and evdev input devices or on many other operating systems, or a nested compositor that itself is an X11 or Wayland application (client). Wayland can even be used in application-internal communication as is done in some web browsers.
Wayland 不限制它的使用地点和方式。Wayland 合成器可以是在 Linux 内核模式设置和 evdev 输入设备或许多其他操作系统上运行的独立显示服务器，也可以是本身是 X11 或 Wayland 应用程序（客户端）的嵌套合成器。Wayland 甚至可以用于应用程序内部通信，就像在某些 Web 浏览器中所做的那样。

# Xwayland
[text](https://man.archlinux.org/man/extra/xorg-xwayland/Xwayland.1.en)

Xwayland is an X server and a Wayland client. It plays the role of a proxy between legacy X clients which do not support the Wayland protocols and the Wayland server.

Xwayland 是一个 X 服务器和一个 Wayland 客户端。它在不支持 Wayland 协议的遗留 X 客户端和 Wayland 服务器之间扮演代理的角色。

# configure Debian 12 system to use Wayland for all components that support it

Yes, you can configure your Debian 12 system to use Wayland for all components that support it. This might help resolve compatibility issues by ensuring everything is consistently running on Wayland without falling back to X11. Here’s how to do it:

### 1. **Ensure GNOME is Running on Wayland**
By default, Debian 12 GNOME should be set to use Wayland unless it’s explicitly disabled. Here’s how to confirm or switch to a Wayland session:

1. **Log Out** of your current GNOME session.
2. **On the login screen**, select your username, and before logging back in, look for a **gear icon** near the Sign In button.
3. Click the gear icon and select **GNOME** or **GNOME on Wayland** (not GNOME on Xorg).
4. Log back in.

If you don’t see the option for **GNOME on Wayland**, you may need to ensure that Wayland is enabled in the GNOME Display Manager (GDM) configuration file.

### 2. **Enable Wayland in GDM (GNOME Display Manager)**
If Wayland is disabled in GDM, follow these steps to re-enable it:

1. Open a terminal and edit the GDM configuration file:
   ```bash
   sudo nano /etc/gdm3/custom.conf
   ```
   
2. Look for the line:
   ```ini
   #WaylandEnable=false
   ```
   
3. Uncomment it and set it to `true`, or remove the line entirely to allow Wayland by default:
   ```ini
   WaylandEnable=true
   ```
   
4. Save and close the file.
5. Restart GDM to apply the changes:
   ```bash
   sudo systemctl restart gdm3
   ```

### 3. **Set Applications to Prefer Wayland**
Most GNOME applications and modern GTK apps should automatically use Wayland if the system is in a Wayland session. However, some applications (especially older or custom ones) may default to X11. Here’s how to ensure that applications favor Wayland:

- **Check environment variables**: Wayland applications typically recognize `WAYLAND_DISPLAY`. To make sure applications use Wayland, you can set this variable globally:

   1. Open the global profile configuration:
      ```bash
      sudo nano /etc/environment
      ```
   2. Add the following line:
      ```bash
      WAYLAND_DISPLAY=wayland-0
      ```
   3. Save and close the file.
   4. Log out and log back in for changes to take effect.

- **Force specific applications to use Wayland**: Some applications (like Firefox and Electron-based apps) have specific flags to prefer Wayland. Here are a few examples:

   - **Firefox**: Run Firefox with:
     ```bash
     MOZ_ENABLE_WAYLAND=1 firefox
     ```

   - **Electron apps**: For Electron apps (like Visual Studio Code), start them with:
     ```bash
     code --enable-features=UseOzonePlatform --ozone-platform=wayland
     ```

### 4. **Verify that Everything is Running on Wayland**
Once you’ve made these changes, you can verify that GNOME and your applications are running on Wayland:

- **Check GNOME session**: Open a terminal and run:
  ```bash
  echo $XDG_SESSION_TYPE
  ```
  If the output is `wayland`, then GNOME is running on Wayland.

- **Check individual applications**: For example, in `gnome-control-center` or Firefox, you can check which display protocol they’re using:
  - **In `gnome-control-center`**: Run `gnome-control-center` from the terminal and see if there are any `GdkX11`-related warnings (which would indicate it’s trying to use X11). If no warnings appear, it’s likely running on Wayland.
  - **In Firefox**: Type `about:support` in the address bar and look for `Window Protocol` under **Graphics**. It should say `Wayland`.

### 5. **Restart the System**
To ensure all components pick up the changes, restart your computer:
```bash
sudo reboot
```

After this, your GNOME session and most applications should fully run on Wayland, which should help avoid issues related to mixed X11 and Wayland components. If you encounter specific applications that still don’t work well with Wayland, you might need to run them with the `GDK_BACKEND=x11` flag temporarily or wait for further compatibility improvements.

## second

Yes, it is possible to configure your system to use **Wayland** for all components, which would resolve the problem where your GNOME desktop environment and applications are conflicting with X11.

If you want to set **Wayland** as the default display protocol for all components on your Debian 12 system, follow these steps:

### 1. **Ensure Wayland is Installed**
First, ensure that Wayland and the necessary components for GNOME are installed. This includes the **Wayland compositor** and the **GNOME Wayland backend**.

Run the following command to check if the necessary Wayland components are installed:

```bash
sudo apt install wayland-protocols gnome-shell wayland
```

### 2. **Ensure GNOME is Running on Wayland**
By default, GNOME should run on **Wayland** on systems that support it, unless there is a configuration forcing it to run on **X11**. You can verify that GNOME is using Wayland by checking the `GDK_BACKEND` environment variable, which should be set to `wayland`.

#### Step-by-Step to Force GNOME to Use Wayland:

1. **Check Current Session**
   To check whether GNOME is running on Wayland or X11, open a terminal and run the following command:
   
   ```bash
   echo $XDG_SESSION_TYPE
   ```

   - If the output is `wayland`, GNOME is already using Wayland.
   - If the output is `x11`, GNOME is using X11, and you’ll need to configure it to use Wayland.

2. **Log Out of the Current Session**
   Log out of your current GNOME session.

3. **Choose GNOME on Wayland (at the Login Screen)**
   - On the login screen, after entering your username, you should see a settings icon or a gear icon (usually in the lower-right corner or near the session selection).
   - Select **GNOME on Wayland**.
   - Then log back in.

This will start GNOME under **Wayland** rather than X11.

### 3. **Disable X11 as a Fallback Option (Optional)**

To make sure that your system does not fall back to X11 automatically, you can disable the ability to use X11 for GNOME.

1. **Edit the GDM Configuration**:
   
   Open the file `/etc/gdm3/custom.conf` in a text editor with superuser privileges. For example, use `nano`:

   ```bash
   sudo nano /etc/gdm3/custom.conf
   ```

2. **Disable Wayland and Ensure X11 is Not Available**:
   
   Find the line:
   ```ini
   #WaylandEnable=false
   ```
   and make sure it is **commented out** (no `#` before it). This ensures that Wayland is enabled for GNOME.

   Now, find the line:
   ```ini
   #WaylandEnable=false
   ```
   and make sure it is **disabled** (no `#` before it). 

3. **Restart GDM (GNOME Display Manager)**:
   
   After making the changes, restart the GDM service (this controls the graphical login):
   
   ```bash
   sudo systemctl restart gdm3
   ```

4. **Reboot Your System**:
   
   Finally, reboot your computer to make sure everything is set properly:
   
   ```bash
   sudo reboot
   ```

### 4. **Verifying the System is Using Wayland**
After logging back in, you can confirm that all components are using Wayland by running:

```bash
echo $XDG_SESSION_TYPE
```

It should output `wayland` to confirm the system is running on Wayland.

### 5. **Check for Potential Problems with Wayland**
While most modern GNOME applications and components should work well under Wayland, there might still be some legacy X11-specific issues or applications that might not fully support Wayland yet. In that case:

- **Fallback to X11 for specific apps**: If you encounter issues with specific applications, you can still run them under X11 by explicitly setting the `GDK_BACKEND=x11` variable for those applications only.

- **System-wide Wayland**: If you want to use Wayland across all applications and ensure that they are compatible with it, consider looking into compatibility layers or tools that support running X11 applications under Wayland. For example, you could use `XWayland`, which allows X11 applications to run on Wayland.

### Conclusion
Once you've configured GNOME to use **Wayland** and ensured that all components are set up to run under it, your system should no longer experience the issues related to X11-compatibility. However, always be prepared for occasional compatibility issues with legacy applications, although this is becoming less of a problem over time as more software supports Wayland natively.