# Parameter estimation for a Li-ion battery cell

## Retrieving the data

Data is available from the [CALCE](https://calce.umd.edu/battery-data) website and can be downloaded and extracted into a `calce/` directory with:

```bash
./get_data.sh
```

Before running the parameter estimation, the incremental discharge data should also be processed into steady-state values:

```bash
python process_ocv.py
```

## Running the script

The script is written in MyST Markdown form so that it can either be converted into a Python script or rendered to HTML for the website using Sphinx.
To convert to a script and execute, run:

```bash
jupytext battery_sysid.md --to py && python battery_sysid.py
```

You may need to `pip install jupytext` if it's not installed in your environment.