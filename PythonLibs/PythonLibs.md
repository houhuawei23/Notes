# Python Libs

## Misc

- pypi tuna source: `https://pypi.tuna.tsinghua.edu.cn/simple`
  - [tuna help](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- anaconda:
  - [tuna help](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

## Misc

### Pip

```bash
# all install pkgs
pip freeze > requirements.txt
# only pkgs used in project
pip install pipreqs
pipreqs path/to/your/project

pip install -r requirements.txt
```

### [Conda](https://anaconda.org/anaconda/conda)

- OS-agnostic, system-level binary package and environment manager.
- 与操作系统无关的系统级二进制包和环境管理器。

- [github: conda](https://github.com/conda/conda)
- [docs](https://docs.conda.io/en/latest/)

Conda is an open source package management system and environment management system for installing multiple versions of software packages and their dependencies and switching easily between them. It works on Linux, OS X and Windows, and was created for Python programs but can package and distribute any software.

Conda 是一个开源包管理系统和环境管理系统，用于安装多个版本的软件包及其依赖项，并在它们之间轻松切换。它适用于 Linux、OS X 和 Windows，是为 Python 程序创建的，但可以打包和分发任何软件。

```bash
# list/update/install/remove/search

conda list
conda list numpy

conda update --all

conda install numpy

# create env
conda create -n env_name list_of_packages

# list envs
conda env list

# activate/deactivate
conda activate/deactivate env_name

# check config info
conda config --show

# change source
conda config --set custom_channels.auto https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
# export
conda env export > environment.yml

# close auto activate
conda config --set auto_activate_base false
```

```bash
# add to tuna source
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

# recover default source
conda config --remove-key channels
```

conda 设置环境变量 [doc](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables)

```bash
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```

#### [Miniconda](https://docs.anaconda.com/miniconda/)

Miniconda 是 Anaconda Distribution 的免费微型安装，其中仅包括 conda、Python、它们都依赖的包以及少量其他有用的包。

如果您需要更多软件包，请使用命令 `conda install` 从 Anaconda 的公共存储库中默认可用的数千个软件包中安装，或者从 `conda-forge` 或 `bioconda` 等其他 channels 进行安装。

### ipython

```bash
# in vscode, ipython extension need ipykernel (manual install):
conda install -n {your_env_name} ipykernel --update-deps --force-reinstall
```

### Module Dependency Check

- `pipdeptree`
- [`pydeps`](https://github.com/thebjorn/pydeps): Python Module Dependency graphs
  - `pydeps xxx.py`

```bash
# show all packages installed
pip list
conda list

# show package details
pip show numpy
conda list numpy

# upgrade package
pip install --upgrade pyarrow
# rebuild dependencies
pip uninstall pyarrow
pip install pyarrow --no-binary pyarrow  # Force recompile

# use pipdeptree to visualize the dep tree
pip install pipdeptree
pipdeptree # show all dependencies
pipdeptree --packages pyarrow # only show pyarrow
pipdeptree --reverse --packages pyarrow # show who dep on pyarrow


```

### [Pylint](https://pylint.pycqa.org/en/latest/index.html): Static Code Analyser

Additional tools in pylint

- [pyreverse](https://pylint.readthedocs.io/en/latest/additional_tools/pyreverse/index.html)(standalone tool that generates package and class diagrams.)
- [symilar](https://pylint.readthedocs.io/en/latest/additional_tools/symilar/index.html)(duplicate code finder that is also integrated in pylint)

#### pyreverse

```bash
pyreverse [options] <packages>

# eg.
# generate classes.dot and packages.dot of marko lib
pyreverse marko 
01_Misc$ dot -Tsvg classes.dot > classes.svg
01_Misc$ dot -Tsvg packages.dot > packages.svg
```
