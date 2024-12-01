[text](https://a76yyyy.github.io/clash/zh_CN/configuration/rules.html)

[text](https://clashhk.com/)

## Clash rules

To ensure that Clash does not redirect traffic to your Tailscale server's hostname and allows Tailscale to handle it, you can write a **`DOMAIN-SUFFIX` rule** in your Clash configuration file (`config.yaml` or any custom configuration file you are using). Here's how you can do it:

### Steps to Update Clash Configuration:

1. **Open your Clash configuration file**:
   Locate your `config.yaml` file or the configuration file you are using for Clash. It is typically found in `~/.config/clash/` or the directory specified during setup.

2. **Add a rule for your Tailscale hostname**:
   Under the `rules:` section, add a rule that bypasses traffic for the specific hostname.

   ```yaml
   rules:
     - DOMAIN-SUFFIX,my_tailscale_server_hostname,DIRECT
   ```

   Replace `my_tailscale_server_hostname` with your actual Tailscale hostname.

3. **Ensure that Clash processes the rule properly**:
   Make sure this rule is placed before any generic rules that might otherwise catch and redirect this traffic.

4. **Restart Clash**:
   After saving the configuration, restart your Clash service to apply the changes:
   ```bash
   systemctl restart clash
   ```

### Explanation:
- `DOMAIN-SUFFIX`: Matches the hostname (or its suffix) you specify.
- `my_tailscale_server_hostname`: Your Tailscale server hostname (e.g., `myserver.example.com`).
- `DIRECT`: Tells Clash to allow traffic to flow directly without interference.

### Additional Considerations:
- If Tailscale uses an IP address rather than a hostname, you can use a rule like this:
  ```yaml
  rules:
    - IP-CIDR,100.64.0.0/10,DIRECT
  ```
  This ensures that all traffic to Tailscale's private network range (`100.64.0.0/10`) is not redirected.

- If your Tailscale hostname resolves dynamically or has multiple domains, you might need additional `DOMAIN-SUFFIX` or `DOMAIN` rules to cover all possibilities.

With this setup, Clash will bypass traffic intended for `my_tailscale_server_hostname:9090`, allowing Tailscale to handle it seamlessly.

## Clash DNS

The error occurs because Clash cannot resolve the domain `xx-rocky` to an IP address. This can happen if:

1. The hostname `xx-rocky` is not resolvable via DNS.
2. It relies on Tailscale's private DNS resolver, which Clash may not be configured to use.

Here’s how to resolve the issue:

---

### Option 1: Add the Tailscale DNS to Clash
Ensure Clash uses the Tailscale DNS resolver. Tailscale often sets up a private DNS resolver to resolve hostnames like `xx-rocky`.

1. Find the Tailscale DNS server:
   Run:
   ```bash
   tailscale ip -4
   ```
   This will give you the IPv4 address of your Tailscale server. Additionally, check the DNS settings from:
   ```bash
   tailscale status
   ```

2. Add the Tailscale DNS server to Clash’s DNS configuration:
   Modify the `dns:` section of your `config.yaml` file:
   ```yaml
   dns:
     enable: true
     listen: 0.0.0.0:53
     nameserver:
       - 100.100.100.100  # Replace with Tailscale DNS server if different
       - 8.8.8.8          # Backup: Google Public DNS
       - 1.1.1.1          # Backup: Cloudflare DNS
     fallback:
       - 8.8.8.8
       - 1.1.1.1
   ```

3. Save the configuration and restart Clash:
   ```bash
   systemctl restart clash
   ```

---

### Option 2: Use IP Address in the Rule
If the hostname `xx-rocky` resolves to a known static IP (or an IP in Tailscale's range), you can write a rule that bypasses based on the IP address instead:

1. Find the IP address of `xx-rocky`:
   Run:
   ```bash
   tailscale ping xx-rocky
   tailscale ip -4
   ```
   This will show the Tailscale-assigned IP for the hostname.

2. Add an `IP-CIDR` rule to bypass traffic for that IP or subnet:
   ```yaml
   rules:
     - IP-CIDR,100.64.0.0/10,DIRECT
   ```

3. Restart Clash:
   ```bash
   systemctl restart clash
   ```

---

### Option 3: Add a Static Host Entry in Clash
If the hostname is not resolvable via DNS, you can manually define it in Clash’s configuration:

1. Add a static mapping in the `hosts:` section of `config.yaml`:
   ```yaml
   dns:
     enable: true
     listen: 0.0.0.0:53
     hosts:
       "xx-rocky": 100.64.0.1  # Replace with the Tailscale IP of xx-rocky
   ```

2. Save the configuration and restart Clash:
   ```bash
   systemctl restart clash
   ```

---

### Option 4: Bypass Clash for Local Traffic
If you prefer to bypass Clash entirely for all Tailscale traffic:

1. Add Tailscale's subnet to Clash's `bypass` list in `config.yaml`:
   ```yaml
   rules:
     - IP-CIDR,100.64.0.0/10,DIRECT
   ```

2. Restart Clash:
   ```bash
   systemctl restart clash
   ```

---

By following these steps, you should resolve the issue and ensure that traffic to `xx-rocky` is properly routed through Tailscale.