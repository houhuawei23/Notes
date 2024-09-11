```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker Engine:
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

https://docs.docker.com/desktop/install/debian/

```bash
docker stop $(docker ps -a -q)
docker image pull name:tag


docker search image_name
docker pull image_name:tag
docker images # list all images
docker rmi image_name:tag # remove image
docker run -it image_name:tag /bin/bash # run container

# start docker desktop
systemctl --user start docker-desktop
# start on sign in
systemctl --user enable docker-desktop
systemctl --user stop docker-desktop
```

https://medium.com/@SrvZ/docker-proxy-and-my-struggles-a4fd6de21861


