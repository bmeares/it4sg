import pandas as pd
import seaborn as sns
from pylab import savefig

filepath = "combinedData.csv"

tze_data = pd.read_csv(filepath)


print(tze_data.head())
print(tze_data.describe())


#timestamp format : 01.01.2017  09:00:00
#year_one = tze_data[:tze_data["timestamp"].loc["01.01.2018 00:00:00"]]
ax = tze_data.plot()
fig = ax.get_figure()
fig.savefig("memes.png")

graph = sns.lineplot(x=tze_data["timestamp"], y=tze_data["Netz_Wirkleistung"])
graph.savefig("fig.png", dpi=600)
