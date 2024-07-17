

Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).

Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

```bash
mkdir .mujoco
cd .mujoco
tar -xzvf mujoco210-macos-x86_64.tar.gz
```

~/.bashrc

```hb
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
```

```bash
source .bashrc

conda activate your_env
pip3 install -U 'mujoco-py<2.2,>=2.1'
```

**Possible errors and solutions**
https://github.com/openai/mujoco-py/issues/627
https://github.com/openai/mujoco-py/issues/773

**Error display shows no opengl**

```bash
apt-get download libopengl0
cd ~
mkdir libopengl0
dpkg -x libopengl0*_amd64.deb libopengl0
```

~/.bashrc

```hf
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/libopengl0/usr/lib/x86_64-linux-gnu
```

```bash
source .bashrc
```
