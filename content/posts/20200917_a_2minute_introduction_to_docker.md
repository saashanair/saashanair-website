---
title: "A 2-minute Introduction to Docker"
date: 2020-09-17
tags: "Misc Tech"
---
Containers are a really neat piece of technology. The first thing to pop into my mind every time I think of containers is an image of tiffin carriers/dabbas (all the South Asian readers here will be able to relate 😋, and for the non-South Asians here, think of meal kits instead 🤪). Similar to how dabbas are popular for allowing you to pack a nutritious, self-contained meal, containers allow you to encapsulate your intended applications and all its dependencies in a self-sufficient system.

<img src="/images/posts/20200917_a_2minute_introduction_to_docker/dabba.png" class="large" alt="Pictures of dabba, meal kit and docker containers">
<em>Dabba (<a href="https://www.maxpixel.net/India-Shiny-Pans-Pan-Lunch-Stainless-Steel-Tiffin-1796184">source</a>) or Meal Kit (<a href="https://medium.com/@giacaglia/the-end-of-the-meal-kit-ffc707a1957">source</a>) vs Docker Containers (<a href="https://medium.com/codingthesmartway-com-blog/docker-beginners-guide-part-1-images-containers-6f3507fffc98">source</a>)</em>

So, what do the terms docker, images and containers mean? Well, to start, Docker is the platform that supports containerisation, i.e. it allows you to create containers to easily run and share your applications. A docker image can be thought of as a snapshot that contains the instructions for setting up the environment to run your applications. A docker image is read-only, and hence cannot be modified. Thus, to actually perform the desired tasks, you need to use the image to create a container. You often don’t build an image from scratch, and instead use one of the images available for use in the online registries ([List of images in the NVIDIA Registry optimised for GPU usage](https://catalog.ngc.nvidia.com)[)](https://ngc.nvidia.com/catalog/all](https://ngc.nvidia.com/catalog/all)). Do note, these explanations are meant to give you a quick overview of the topics, and are not the exact/precise definitions used in "industry speak".

All this is well and good, but why am I talking about containers today out of nowhere, you might wonder. It is after all a departure from the normal topics revolving around AI. For context, I finally started in-presence work here in Italy this month, and have been granted access to the NVIDIA DGX system. Though having the opportunity to run experiments on a powerful system sounds amazing, it also meant I had to get familiar with Docker. Last Thursday I finally took the plunge, and wanted to give you a list to hit the ground running so that you don't have to spend hours asking Google for answers.

Aside: Yes, I managed to shift to Italy, despite the pandemic! 🎉 Sorting out paperwork in the two countries has been an ordeal, but oh well! Looking forward to the new experiences that Italy has in store 💙)!

## **A list of commands to help you get your hands dirty**

1. **List all locally available docker images:**
    
    ```bash
    docker image ls
    ```

    <img src="/images/posts/20200917_a_2minute_introduction_to_docker/result1.png" class="large" alt="Result">

1. **Pull the desired docker image:**
    
    ```bash
    docker pull <image_name>:<image_tag>
    ```
    
    For example, to pull the NVIDIA PyTorch container (available at: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch),
    
    ```bash
    docker pull nvcr.io/nvidia/pytorch:20.08-py3
    ```
    
    Here, [nvcr.io/nvidia/pytorch](http://nvcr.io/nvidia/pytorch:20.08-py3) is the name of the container and [20.08-py3](http://nvcr.io/nvidia/pytorch:20.08-py3) is the tag.

    <img src="/images/posts/20200917_a_2minute_introduction_to_docker/result2.png" class="large" alt="Result">

1. **List the details of all docker containers on your system, along with their status**
    
    ```bash
    docker ps -a
    ```
    
    <img src="/images/posts/20200917_a_2minute_introduction_to_docker/result3.png" class="large" alt="Result">
    
    Viewing the entire list can sometimes seem overwhelming, thus, to view only the latest container, replace the -a in the above command with -l. It thus becomes, `docker ps -l`
    

1. **List only the currently running docker containers**
    
    ```bash
     docker container ls
    ```
    
    <img src="/images/posts/20200917_a_2minute_introduction_to_docker/result4.png" class="large" alt="Result">

1. **Create/Run a docker container**
    
    ```bash
    docker run --gpus all -it --rm -v <local_dir>:<container_dir> <image_name>:<image_tag>
    ```
    
    The -it option starts the container in interactive mode, while --rm is used to ensure that the container is deleted from the system as soon as you exit out of it. The -v option is used to mount a volume to the container, which can be useful for accessing your datasets that need to be used for training. Remember though that both the local_dir and container_dir need to be absolute path. For example, using the previously downloaded Pytorch image, the container could be created as below,
    
    ```bash
    docker run --gpus all -it --rm -v /data/:/container_data/ nvcr.io/nvidia/pytorch:20.08-py3
    ```
    

1. **Create a named docker container that persists even after you exit the terminal**
    
    There are a two options for this:
    
    a) Use the same run command as above, but without the --rm option. This starts the container in interactive mode, and when you type "exit", it shuts down the container. To use the container again, you need to start the container (see point 7 below) and access the terminal (see point 9 below).
    
    ```bash
    docker run --gpus all -it --name <name> -v <local_dir>:<container_dir> <image_name>:<image_tag>
    ```
    
    b) Use the create command which makes a new container, and sets the status to "Created". Following this, similar to the option (a) above, you need to start the container (see point 7 below) and then access the terminal (see point 9 below). For some reason, while using this option, I had to use the start command twice, for the container to be started.
    
    ```bash
    docker create --gpus all --name <name> -v <local_dir>:<container_dir> <image_name>:<image_tag>
    ```
    
    *Note:* If you do not want to name the container, and don’t mind the name that docker automatically sets, you could skip the ‘-n’ option.
    

1. **Start a docker container**
    
    ```bash
    docker start <container_name>
    ```
    
    If you used the -n option while creating the container, then you can directly use the name that you set. But if not, use the `docker ps -a` or `docker ps -l` command to get the name set to the container by docker.
    
    <img src="/images/posts/20200917_a_2minute_introduction_to_docker/result5.png" class="large" alt="Result">
    

1. **Stop a docker container**
    
    ```bash
    docker stop <container_name>
    ```
    
    <img src="/images/posts/20200917_a_2minute_introduction_to_docker/result6.png" class="large" alt="Result">

1. **Access the terminal of the docker container in interactive mode**
    
    ```bash
    docker exec -it <container_name> /bin/bash
    ```
    
    You can exit the terminal, while allowing the container to continue running by simply typing `exit` at the container's terminal.
    
    <img src="/images/posts/20200917_a_2minute_introduction_to_docker/result7.png" class="large" alt="Result">
    

1. **Get the IP address of a docker container**
    
    
    ```bash
    docker inspect -f " {{ .NetworkSettings.IPAddress }} " <container_name>
    ```
    
2. **Delete a docker stopped/exited container**
    
    ```bash
    docker rm <container_name>
    ```
    
    <img src="/images/posts/20200917_a_2minute_introduction_to_docker/result8.png" class="large" alt="Result">


Hope this article helped you. If you have suggestions for any other commands that you feel everyone should add to their toolbox, drop me a message at [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com), or hit me up on [Twitter](https://twitter.com/saasha_nair). 💙

See you in the next post dear reader. Have a nice day! ☀️

---

## **Resources for further reading:**

- [Official Docker Documentation](https://docs.docker.com/get-started/)
- [NVIDIA GPU Cloud Documentation](https://docs.nvidia.com/ngc/ngc-user-guide/nvcontainers.html#nvcontainers)
- [List of the NVIDIA Image Registry](https://ngc.nvidia.com/catalog/all)