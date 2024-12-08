# ADMIRE Vision-Language Model Finetuning

### Setup
- Install Pyenv: https://github.com/pyenv/pyenv
- Install the Python version specified in `.python-version` with Pyenv: `pyenv install 3.12.6`
- Create virtual environment: `python -m venv venv`
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

If you choose to download additional packages (eg `seaborn`), do the following steps:
1. `pip install seaborn`
2. `pip freeze > requirements.txt`

If pre-commit hooks don't seem to be working, try the following:
1. `pre-commit install`
2. `pre-commit run --all-files`
This ensures that everyone has access the same pinned, resolved versions of dependencies.


### Code

Hey Sicong. All of the code that we actually used is in the `/v2` folder. The stuff outside the v2 folder (top-level) were failed attempts to use Unsloth. I ended up just using TRL.
Edit: I cut a bunch of files from the top-level that ended up not being relevant. A lot of work there, but nothing that made it into the final finetune/eval.

Note: the other part of this project's code submission is the Synthetic Data pipeline here: https://github.com/sasilver75/admire-pipeline
