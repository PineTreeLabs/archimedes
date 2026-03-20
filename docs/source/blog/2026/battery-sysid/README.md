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

