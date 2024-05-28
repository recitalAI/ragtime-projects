PROJECT_NAME: str = "Albert"

import ragtime
from ragtime.pipeline import Configuration, reference_LLM
import ragtime.pipeline
from albert_llm_api_caller import Albert_LLM
import yaml

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")

# Here we set an internal ragtime state to register our custom LLM classe
reference_LLM(Albert_LLM, "Albert_LLM")

# Then we load the config file
configuration: dict = dict()

with open("conf.yaml") as file:
    configuration = yaml.load(file.read())

# Instantiate the pipeline and run it
pipline = Configuration(**configuration)
pipline.run()
