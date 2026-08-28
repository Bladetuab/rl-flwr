import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("convlstm_anomaly.csv")

agg = df.groupby("client_id").agg(
    mean_score=("mean_score", "mean"),
    mean_var=("var_score", "mean"),
    is_malicious=("is_malicious", "max"),
).reset_index()

benign = agg[agg.is_malicious == 0]
mal = agg[agg.is_malicious == 1]

plt.figure()
plt.scatter(benign.mean_score, benign.mean_var, label="benign", alpha=0.7)
plt.scatter(mal.mean_score, mal.mean_var, label="malicious", alpha=0.7)
plt.xlabel("Mean ConvLSTM anomaly score")
plt.ylabel("Mean bootstrap variance")
plt.legend()
plt.title("Benign vs Malicious Clients (ConvLSTM + Bootstrapping)")
plt.show()
