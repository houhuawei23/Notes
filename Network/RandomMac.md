To use a random MAC address on Debian 12, you can utilize a few different methods, including temporary changes or setting it to generate random addresses automatically. Here's how you can do it:

### Method 1: Temporary Change Using `macchanger`

1. **Install `macchanger`**:
   Open a terminal and install `macchanger` using the following command:

   ```bash
   sudo apt update
   sudo apt install macchanger
   ```

2. **Identify Your Network Interface**:
   Find the name of your network interface (e.g., `wlan0`, `eth0`) by running:

   ```bash
   ip link
   ```

3. **Bring Down the Interface**:
   Replace `wlan0` with your interface name:

   ```bash
   sudo ip link set wlan0 down
   ```

4. **Change the MAC Address**:
   Use `macchanger` to set a random MAC address:

   ```bash
   sudo macchanger -r wlan0
   ```

5. **Bring Up the Interface**:

   ```bash
   sudo ip link set wlan0 up
   ```

6. **Verify the Change**:

   ```bash
   ip link show wlan0
   ```

### Method 2: Permanent Random MAC Address on Startup

To have a random MAC address every time you boot, you can modify the Network Manager configuration.

1. **Open NetworkManager Configuration**:
   Edit the connection file for your network interface. The path may vary, but you can typically find it in `/etc/NetworkManager/system-connections/`. You can list the available connections with:

   ```bash
   sudo ls /etc/NetworkManager/system-connections/
   ```

   Then open the specific connection file using a text editor (replace `YourConnection` with the actual connection name):

   ```bash
   sudo nano /etc/NetworkManager/system-connections/YourConnection
   ```

2. **Modify the Configuration**:
   Look for the `[802-11-wireless]` section and add or modify the following lines:

   ```plaintext
   [connection]
   ...
   ethernet.cloned-mac-address=random
   ```

   or for Wi-Fi connections:

   ```plaintext
   [802-11-wireless]
   ...
   802-11-wireless.cloned-mac-address=random
   ```

3. **Save and Exit**:
   Save the file and exit the editor.

4. **Restart NetworkManager**:
   Restart the NetworkManager to apply the changes:

   ```bash
   sudo systemctl restart NetworkManager
   ```

### Method 3: Manual Random MAC Address Generation

If you want to generate a random MAC address manually, you can use a simple script or command.

1. **Generate a Random MAC Address**:
   You can use the following command to generate a random MAC address:

   ```bash
   printf '02:%x:%x:%x:%x:%x:%x\n' $((RANDOM%256)) $((RANDOM%256)) $((RANDOM%256)) $((RANDOM%256)) $((RANDOM%256)) $((RANDOM%256))
   ```

2. **Set the Random MAC Address**:
   Use `ip` to change the MAC address as shown in Method 1, substituting `XX:XX:XX:XX:XX:XX` with the output of the above command.

### Notes

- **Temporary Changes**: The MAC address will revert to the original on reboot unless you use the NetworkManager method.
- **Network Policies**: Be aware that changing your MAC address may violate certain network policies.
- **Conflict Avoidance**: Ensure that the generated MAC address does not conflict with other devices on your network.

By following these methods, you can effectively use a random MAC address on your Debian 12 system!