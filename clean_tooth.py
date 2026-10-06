# -*- coding: utf-8 -*-
"""清洗 tooth24: sex: Boys=0, 0.0=Girls=1; dmf: Sound=0, 0.0=dmf阳性=1."""
import pandas as pd, numpy as np

df = pd.read_csv(r"G:\OPT\Weibull发表策略\data\tooth24.csv")
df['girl'] = (df['sex'].astype(str).str.strip() == '0.0').astype(int)  # 0.0=Girls
df['dmf_pos'] = (df['dmf'].astype(str).str.strip() == '0.0').astype(int)  # 0.0=dmf阳性
df['right'] = df['right'].replace([np.inf], np.nan)
df['censored'] = df['right'].isna().astype(int)

out = df[['id','left','right','censored','girl','dmf_pos']].dropna(subset=['left'])
out.to_csv(r"G:\OPT\Weibull发表策略\data\tooth24_clean.csv", index=False)

msg = (f"shape={out.shape}\n"
       f"interval={(out.censored==0).sum()} right_censored={out.censored.sum()}\n"
       f"left=[{out.left.min():.2f},{out.left.max():.2f}]\n"
       f"right=[{out.right.min():.2f},{out.right.max():.2f}]\n"
       f"girls={out.girl.mean()*100:.1f}%\n"
       f"dmf_pos={out.dmf_pos.mean()*100:.1f}%\n"
       f"by_girl: interval={out[out.girl==1].censored.eq(0).sum()}, rcens={out[out.girl==1].censored.sum()}\n"
       f"by_boy: interval={out[out.girl==0].censored.eq(0).sum()}, rcens={out[out.girl==0].censored.sum()}")
open(r"G:\OPT\Weibull发表策略\data\summary.txt","w").write(msg)
