import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("P:\\Prodigy\\Task-02\\train.csv")

df.head()
df.info()
df.describe()


df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)


df.drop(columns=["Cabin"], inplace=True)


df["Survived"].value_counts().plot(kind="bar")
plt.xlabel("Survival (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.title("Survival Count")
plt.show()


pd.crosstab(df["Sex"], df["Survived"]).plot(kind="bar")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()


pd.crosstab(df["Pclass"], df["Survived"]).plot(kind="bar")
plt.title("Survival by Passenger Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


plt.hist(df["Age"], bins=30)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution of Passengers")
plt.show()


