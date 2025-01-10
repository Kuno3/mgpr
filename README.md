# mgpr

単調制約付きのガウス過程回帰のパッケージ

## ガウス過程回帰の使い方

```Python
model = gpr(
    t,
    y,
    sigma2=1.0,
    rho2=1.0,
    gamma2=1.0
) # tが観測値の入力でyが観測値の出力．sigma2とrho2，gamma2がパラメータ
model.parameter_optimize() # パラメータの最適化（初期値は最初に与えたパラメータ）
samples = model.sampling(s, num_samples) # sは確率密度関数に興味のある点の入力でnum_samplesは欲しいサンプルの数
```

## 単調制約付きガウス過程回帰の使い方

```Python
model = mgpr(
    t,
    y,
    x,
    sigma2=1.0,
    rho2=1.0,
    gamma2=1.0
) # tが観測値の入力でyが観測値の出力．sigma2とrho2，gamma2がパラメータ
# xは単調制約を入れる点だが，点が細かすぎたり多すぎると失敗する．30点以下を推奨
model.parameter_optimize() # パラメータの最適化（初期値は最初に与えたパラメータ）
# 最適化はscipyのminimizeによって行っており，頻繁に失敗するため，初期値を色々と与えて試すことを推奨
samples = model.sampling(s, num_samples) # sは確率密度関数に興味のある点の入力でnum_samplesは欲しいサンプルの数
```
