# Presentation
The `ragtime-projects` repo contains projects built using **Ragtime** 🎹.

# Getting started
## Install the package
Make sure you have already installed the Ragtime 🎹 package. If not, just run:
```shell
pip install ragtime
```

If you want to modify the `ragtime-package` then you will want to install it as an editable version. In this case, go to the base folder of `ragtime-package` and run in the shell:
```
pip install -e .
```
(don't forget the last dot).
If you have an error `(A "pyproject.toml" file was found, but editable mode currently requires a setup.py based build.)`, upgrade `pip`.

## Create your project
Create a folder where you will create your Ragtime 🎹 projects. In this folder, create a `main.py` file containing:
```
PROJECT_NAME:str = "your-project-name"

from ragtime import config
config.init_project(name=PROJECT_NAME, init_type='copy_base_files')
```
Set the `PROJECT_NAME` variable with your project name and run the script.

This will create a subfolder named according to `PROJECT_NAME` containing:
- folder `expe`: contains 4 sub-folders containing the data which will be created at each step of your experiments
- folder `logs`: the logs associated with your project
- folder `res`: contains templates for the file exports
- `classes.py`: your classes to define the Prompter and Retrievers you may use
- `main.py`: the file containing the `main` function for your project
- `LICENSE`: MIT by default - don't forget to add you name / company in it
- `README.md`: your project's doc
- `.gitignore`

# Examples
Several examples are given to illustrate how to use Ragtime 🎹:
- [What do LLM think?](what_do_LLM_think/README.md)
- [Google NQ](google_nq/README.md)

# Contributing
Glad you wish to contribute! More details [here](CONTRIBUTING.md).

# Package repo
If you are looking for the package, not the examples, [it is here](https://github.com/recitalAI/ragtime-package).

# Troubleshooting
## Environment variables for API not recognized in Windows
In this case, just call `ragtime.config.init_win_env' as detailed [here](https://github.com/recitalAI/ragtime-package/tree/main?tab=readme-ov-file#windows)