import numpy as np
import pandas as pd

# ==== Incremental OCV Test ====

df1 = pd.read_excel("12_2_2015_Incremental OCV test_SP20-1.xlsx", sheet_name="Channel_1-005_1", usecols="B,G:J")
df2 = pd.read_excel("12_2_2015_Incremental OCV test_SP20-1.xlsx", sheet_name="Channel_1-005_2", usecols="B,G:J")
df3 = pd.read_excel("12_2_2015_Incremental OCV test_SP20-1.xlsx", sheet_name="Channel_1-005_3", usecols="B,G:J")

df = pd.concat([df1, df2, df3])
df = df[df.index > 100].reset_index(drop=True)

# Chunk the df up by charge/discharge cycles
C = 2.0  # Battery capacity in Ah
current = np.array(df["Current(A)"])
voltage = np.array(df["Voltage(V)"])
time = np.array(df["Test_Time(s)"])
soc = C + np.cumsum(current[:-1] * np.diff(time) / 3600)
soc = np.append(soc, soc[-1]) * 100 / C
df["SOC(%)"] = pd.Series(soc, index=df.index)
df["Current(A)"] = -current  # Positive for discharge

discharge = np.array(current < -C/4, dtype=float)
discharge_start = 1 + np.nonzero(np.diff(discharge) > 0)[0]
discharge_end = 1 + np.nonzero(np.diff(discharge) < 0)[0]

charge = np.array(current > C/4, dtype=float)
charge_start = 1 + np.nonzero(np.diff(charge) > 0)[0]
charge_end = 1 + np.nonzero(np.diff(charge) < 0)[0]

# Indices of steady-state points (just before charge/discharge starts)
steady_idx = np.concatenate([discharge_start - 1, charge_start - 1])

soc_steady = soc[steady_idx]
ocv_steady = voltage[steady_idx]

# Save to CSV
df_out = pd.DataFrame({
    "SOC": soc_steady,
    "OCV": ocv_steady,
})
df_out.to_csv("SOC-OCV.csv", index=False)
