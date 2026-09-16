# Kochi Whale Shark Risk Monitor

高知県沿岸の定置網におけるジンベエザメ出現リスクを、現在・予報・履歴として表示するStreamlitアプリです。

## v0.5の主な変更

- 20地点の水深をGEBCOから **J-EGG500のIDW水深**へ更新
- NetID 7・14は漁業権ポリゴン照合後の補正座標、NetID 13は元座標を使用
- 主モデルをR `mgcv` のGAMへ統一
- 水温は生SSTではなく **SST偏差**を使用
- アプリの値を絶対確率ではなく、固定学習データに対する **相対順位**として表示

主モデル：

```r
presence ~
  s(Jday, bs = "cc", k = 20) +
  s(SST_anomaly, k = 10) +
  s(depth_m, k = 10)
```

推定は `family = binomial`, `method = "REML"`, `select = TRUE`。`depth_m` はJ-EGG500 IDW水深です。

## 検証済みの再解析結果

| 指標 | J-EGG500主モデル |
|---|---:|
| 見かけのAUC | 0.904 |
| LONO AUC | 0.858 ± 0.075 |
| AIC | 1349.0 |
| 逸脱度説明率 | 34.7% |
| Jday効果の最大 | 約164日（6月13日） |
| 水深効果の最大 | 約29.4 m |

生SSTはJdayとのconcurvityが高かったため主モデルには採用していません。Jdayを除いた感度解析では、生SSTの最適域は約23.2℃でした。

## データと更新

- `data/net_depth_jegg500.csv`: 座標、旧GEBCO水深、J-EGG500水深、品質区分
- `scripts/model_core.R`: 学習、SST偏差計算、GAM、相対順位への変換
- `scripts/predict_forecast_risk.R`: 現在・予報
- `scripts/predict_historical_risk.R`: 日別・月別・年別履歴
- `.github/workflows/daily_update.yml`: 日次更新

アプリの相対順位は、固定した学習用presence/backgroundデータでのモデル予測値の経験分布に基づきます。**出現確率そのものではありません。**
