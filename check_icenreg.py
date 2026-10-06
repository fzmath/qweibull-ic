import pyreadr, pandas as pd
for f in ["IR_diabetes", "miceData"]:
    d = pyreadr.read_r(rf"G:\OPT\Weibull发表策略\data\icenReg\data\{f}.RData")
    for k,v in d.items():
        print(f"\n=== {f} ===")
        print(v.head(10))
        print(f"shape: {v.shape}")
        print(f"columns: {list(v.columns)}")
