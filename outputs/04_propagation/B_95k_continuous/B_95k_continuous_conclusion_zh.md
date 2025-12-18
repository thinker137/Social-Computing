# 數據集 B_95k_continuous 的傳播／引流代理特徵分析結論草稿

- 總樣本數：100000，spam 佔比約 5.00%。
- 特徵 `no_retweets`：spam 的平均值約為 non-spam 的 0.05 倍，單特徵 ROC-AUC 約為 0.410。
- 特徵 `no_urls`：spam 的平均值約為 non-spam 的 0.96 倍，單特徵 ROC-AUC 約為 0.488。
- 特徵 `no_usermention`：spam 的平均值約為 non-spam 的 0.11 倍，單特徵 ROC-AUC 約為 0.361。

說明：上述三個特徵分別可視為「傳播強度（no_retweets）」「引流強度（no_urls）」與「社交牽引（no_usermention）」的統計代理；本分析基於統計分布與區分能力，屬於**統計層面的傳播模式**，
而非基於具體轉發鏈路或級聯樹的網路結構分析。