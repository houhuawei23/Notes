```bash
# /etc/apt/apt.conf.d/01proxy
# /etc/apt/apt.conf 
Acquire::http::proxy "http://user:password@host:port/";

Acquire {
  HTTP::proxy "http://proxy_server:port/";
  HTTPS::proxy "http://proxy_server:port/";
}

apt search --names-only <package_name>
```