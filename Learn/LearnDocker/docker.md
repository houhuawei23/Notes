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

## Docker Compose Docker 组合

Docker Compose is a tool for defining and running multi-container applications. It is the key to unlocking a streamlined and efficient development and deployment experience.  

Docker Compose 是一个用于定义和运行多容器应用程序的工具。它是解锁精简高效的开发和部署体验的关键。

Compose simplifies the control of your entire application stack, making it easy to manage services, networks, and volumes in a single, comprehensible YAML configuration file. Then, with a single command, you create and start all the services from your configuration file.  

Compose 简化了对整个应用程序堆栈的控制，使您可以在单个易于理解的 YAML 配置文件中轻松管理服务、网络和卷。然后，使用单个命令，您可以从配置文件创建并启动所有服务。

Compose works in all environments; production, staging, development, testing, as well as CI workflows. It also has commands for managing the whole lifecycle of your application:  

Compose 适用于所有环境；生产、登台、开发、测试以及 CI 工作流程。它还具有用于管理应用程序整个生命周期的命令：

- Start, stop, and rebuild services
  - 启动、停止和重建服务
- View the status of running services
  - 查看正在运行的服务的状态
- Stream the log output of running services
  - 流式传输正在运行的服务的日志输出
- Run a one-off command on a service
  - 在服务上运行一次性命令
