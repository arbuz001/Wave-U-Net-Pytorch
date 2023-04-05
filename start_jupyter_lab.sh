#!/usr/bin/env bash
# https://towardsdatascience.com/how-to-connect-to-jupyterlab-remotely-9180b57c45bb
# https://medium.com/mlearning-ai/set-up-remote-jupyter-lab-notebook-server-for-remote-browser-access-2cef464f203e
# steps to launch jupyter lab on remote server
# pip install jupyterlab
# jupyter lab --generate-config
# jupyter lab password
# pwd = mlworkst@tion
# to create config
# nano nano /home/aarbuzov/.jupyter/jupyter_server_config.json
# {  
  # "NotebookApp": {
    # "allow_remote_access":true,
    # "ip":"192.168.112.60",
    # "port":8888,
    # "allow_root":true
  # }
# }


jupyter lab --no-browser --notebook-dir /home/aarbuzov/work-and-study/git/Wave-U-Net-Pytorch/ 
