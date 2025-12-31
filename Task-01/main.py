import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv",
    skiprows=4
)

print(df.head())

population_2020 = df["2020"].dropna()

# plt.hist(population_2020, bins=50, color='blue', edgecolor='black')
# plt.title("Distribution of World Population in 2020")
# plt.xlabel("Population")
# plt.ylabel("Number of Countries")
# plt.grid(axis='y', alpha=0.75)
# plt.show()
# plt.savefig("population_distribution_2020.png")

plt.figure(figsize=(10, 6))

plt.hist(
    population_2020,
    bins=30
)

plt.xlabel("Population")
plt.ylabel("Number of Countries")
plt.title("Distribution of Population Across Countries (2020)")

plt.show()

top10 = df[["Country Name", "2020"]] \
            .dropna() \
            .sort_values(by="2020", ascending=False) \
            .head(10)

plt.figure(figsize=(10, 6))

plt.bar(top10["Country Name"], top10["2020"])

plt.xlabel("Country")
plt.ylabel("Population")
plt.title("Top 10 Most Populous Countries (2020)")

plt.xticks(rotation=45)
plt.show()



