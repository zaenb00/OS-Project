# RL_Schedular

## Running modules ✅

Most scripts in `src/` are intended to be run as *modules* from the project root. For example:

```bash
python -m src.schedulers.fcfs
```

Running a `.py` file directly (for example `python src/schedulers/fcfs.py`) can cause imports to fail because Python won't treat `src` as a package. If you prefer, you can also add `src` to `PYTHONPATH` or install the package in editable mode:

```bash
# PowerShell
$env:PYTHONPATH = "C:\path\to\RL_Schedular\src"
python -m src.schedulers.fcfs

# Or install editable
pip install -e .
```
