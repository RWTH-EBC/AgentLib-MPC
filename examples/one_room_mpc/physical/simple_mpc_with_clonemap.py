import logging
from pathlib import Path
import json
import requests


logger = logging.getLogger(__name__)


def run_example_clonemap():
    # set up full example using regular clonemap
    URL = "http://localhost:30009/api/clonemap/mas"
    CFG_PATH = Path(__file__).parent.joinpath("simple_mpc_clonemap_config.json")
    with open(CFG_PATH, "r") as file:
        DATA = json.load(file)
    requests.post(URL, json=DATA)

    return 0


if __name__ == "__main__":
    run_example_clonemap()
