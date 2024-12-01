
Greate Article for DNS Server Config:

[The Sisyphean Task Of DNS Client Config on Linux](https://tailscale.com/blog/sisyphean-dns-client-linux)

DNS (Domain Name Service): turn names into IP addresses

TimeLine/History/Development:

4.3 BSD: use /etc/resolv.conf to configure DNS servers

but DHCP program need to determine the DNS servers in the network, it may edit and change the /etc/resolv.conf file. When more than one program what to edit the /etc/resolv.conf file, it may cause conflict and cause problems.

resolvconf: a loose convention for managing DNS, common 2 versions are Deian and Openresolv.