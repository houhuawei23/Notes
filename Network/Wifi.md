In Debian 12's Wi-Fi identity management, here are the key terms and their meanings:

1. **SSID (Service Set Identifier)**: This is the name of the Wi-Fi network. It's what you see when you scan for available networks. Each SSID is unique within a given area.

2. **BSSID (Basic Service Set Identifier)**: This refers to the MAC address of the access point (AP) providing the network. While SSID is the name, BSSID identifies the actual hardware of the AP.

3. **MAC Address**: The Media Access Control (MAC) address is a unique identifier assigned to network interfaces for communications on a network. Each device on a network has a distinct MAC address.

4. **Cloned Address Options**:
   - **Preserve**: This option keeps the device's original MAC address.
   - **Permanent**: This sets a fixed MAC address for the device, which will remain the same across reboots and disconnections.
   - **Random**: This generates a new, random MAC address each time the device connects to the network, enhancing privacy by making tracking more difficult.
   - **Stable**: This generates a MAC address that remains consistent but is not the original. It's typically based on the device's original MAC but altered to provide some level of anonymity.

These options help manage network identity and privacy in various scenarios.

在 Debian 12 的 Wi-Fi 身份管理中，以下是关键术语及其含义：

1. **SSID（服务集标识符）**：这是 Wi-Fi 网络的名称。当你扫描可用网络时，会看到这个名称。每个 SSID 在特定区域内是唯一的。

2. **BSSID（基本服务集标识符）**：这指的是提供网络的接入点（AP）的 MAC 地址。虽然 SSID 是名称，但 BSSID 识别的是实际硬件的接入点。

3. **MAC 地址**：媒体访问控制（MAC）地址是分配给网络接口的唯一标识符，用于网络通信。网络上的每个设备都有一个独特的 MAC 地址。

4. **克隆地址选项**：
   - **保留**：此选项保持设备的原始 MAC 地址。
   - **永久**：这会为设备设置一个固定的 MAC 地址，该地址在重启和断开连接后将保持不变。
   - **随机**：这会在设备连接到网络时生成一个新的随机 MAC 地址，提高隐私性，使追踪更困难。
   - **稳定**：这会生成一个保持一致的 MAC 地址，但不是原始地址。通常是基于设备的原始 MAC 地址但进行了修改，以提供一定程度的匿名性。

这些选项有助于在不同场景中管理网络身份和隐私。