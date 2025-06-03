import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"Car Sales.xlsx - car_data.csv")

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Engine'] = df['Engine'].str.replace("Â", "", regex=False)
df.drop_duplicates(inplace=True)


plt.figure(figsize=(10, 6))
top_models = df['Model'].value_counts().nlargest(10)
sns.barplot(x=top_models.values, y=top_models.index, palette="Blues_d")
plt.title("Top 10 Most Sold Car Models")
plt.xlabel("Number of Sales")
plt.ylabel("Car Model")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
avg_price_by_company = df.groupby('Company')['Price ($)'].mean().nlargest(10)
sns.barplot(x=avg_price_by_company.values, y=avg_price_by_company.index, palette="Greens_d")
plt.title("Top 10 Companies by Average Car Price")
plt.xlabel("Average Price ($)")
plt.ylabel("Company")
plt.tight_layout()
plt.show()

sales_by_region_gender = df.groupby(['Dealer_Region', 'Gender'])['Car_id'].count().unstack().fillna(0)
sales_by_region_gender.plot(kind='bar', stacked=True, colormap='Set2', figsize=(14, 7))
plt.title("Car Sales Distribution by Region and Gender")
plt.xlabel("Dealer Region")
plt.ylabel("Number of Cars Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
