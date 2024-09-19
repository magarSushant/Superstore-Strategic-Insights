#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING RELEVANT LIBRARIES

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#IMPORTING THE CSV FILE (SUPERSTORE DATASET)
df = pd.read_csv("C:\\Users\\magar\\OneDrive\\Desktop\\RHUL\\Assignments\\Businss Analytics Language\\Sample - Superstore.csv",encoding='ISO-8859-1')


# In[3]:


#UNDERSTANDING THE CHARACTERISTICS OF THE GIVEN DATA SET
df.head() #Displaying the first few rows of the DataFrame to get a quick overview.


# In[4]:


df.info() # Retrieving information about the dataset, including data types and memory usage.


# In[5]:


df.shape # Determining the number of rows and columns in the DataFrame.


# In[6]:


df.columns # Accessing the names of columns in the DataFrame.


# In[7]:


#DATA PREPROCESSING
df.isnull().sum() # Identifying missing values in a DataFrame


# In[8]:


df.dropna(inplace=True) #Drops/removes the entire row with a null value. Not applicable here since there are no null values.


# In[9]:


df.duplicated() #Identifying duplicate rows


# In[10]:


df.duplicated().value_counts()  #Identifying duplicate rows in terms of count


# In[11]:


# Row ID column serves as a unique identifier and can prevent the detection of duplicates.
df.drop('Row ID', axis=1, inplace=True) #Dropping Row ID which is preventing the identification of duplicates.


# In[12]:


df.head() #Checking if the required column has been dropped or not.


# In[13]:


df.duplicated().value_counts() #Identifying the duplicates again after dropping Row ID


# In[14]:


df[df.duplicated()] #Identifying the exact duplicate


# In[15]:


df.drop(index = 3406, inplace=True) #Dropping the duplicate row.


# In[16]:


df.duplicated().value_counts() #Checking to see the duplicate row is dropped.


# In[17]:


#Domain knowledge-based feature selection
# List of columns to be dropped
columns_to_drop = ['Customer ID', 'Country', 'Postal Code']
# Since Customer Name likely serves the same purpose
# All data is from the United States, the "Country" column does not add any variability or useful information to the analysis.
# Our analysis does not require geographical granularity at the postal code level and have "State" information serves same purpose.

# Drop the specified columns
df.drop(columns=columns_to_drop, inplace=True)


# In[18]:


# Adding the additional unit metrics Columns to the data set.
# Unit metrics facilitate granular insights into profitability, allowing for targeted optimizations and strategic decisions, crucial for effective data analysis and informed business strategies in a dynamic market environment.
df['Unit Discounted Sale Price'] = df['Sales'] / df['Quantity']
df['Unit Original Sale Price'] = df['Unit Discounted Sale Price'] + df['Unit Discounted Sale Price']*df['Discount']
df['Unit Profit']=df['Profit']/df['Quantity']
df['Unit Cost Price']=df['Unit Discounted Sale Price']-df['Unit Profit']
df['Profit Margin']=df['Profit']/df['Sales']*100


# In[19]:


df.head() #Checking the changes made to the dataset.


# In[20]:


#Renaming Column names to make column names more descriptive.
df = df.rename(columns={'Segment': 'Customer Segment'})
df = df.rename(columns={'Category': 'Product Category'})
df = df.rename(columns={'Sub-Category': 'Product Sub-Category'})
df.head()


# In[21]:


#Displaying some of the important KPIs 
# Calculating KPIs
total_sales = df['Sales'].sum()
total_orders = df['Order ID'].nunique()
total_quantity_sold = df['Quantity'].sum()
total_profit = df['Profit'].sum()
profit_margin = (total_profit / total_sales) * 100
number_of_customers = df['Customer Name'].nunique()

# Preparing data for visualization
print("Total Sales:",total_sales)
print("Total Quantity Sold:",total_quantity_sold)
print("Profit Margin:", profit_margin)
print("Total Profit:",total_profit)
print("Total customers:",number_of_customers)
print("Total orders:",total_orders)

KPIs = {'KPI': ['Total Sales', 'Total Quantity Sold', 'Total Profit','Profit Margin', 'Total Customers', 'Total Orders'],
    'Value': [total_sales, total_quantity_sold, total_profit,profit_margin, number_of_customers, total_orders]}

kpi_df = pd.DataFrame(KPIs)


# In[22]:


# Displaying some of the important KPIs 
# Calculating KPIs

average_discount_rate = df['Discount'].mean()
average_quantity_per_order = df['Quantity'].sum() / df['Order ID'].nunique()
average_sales_per_order = df['Sales'].sum() / df['Order ID'].nunique()                                                  
average_profit_per_order = df['Profit'].sum() / df['Order ID'].nunique()


# Preparing data for visualization
print("Average Discount Rate:", average_discount_rate)
print("Average Quantity per Order:", average_quantity_per_order)
print("Average Sales per Order:", average_sales_per_order)
print("Average Profit per Order:", average_profit_per_order)

KPIs_1 = {'KPI': ['Average Discount Rate', 'Average Quantity per Order', 'Average Sales per Order', 'Average Profit per Order'],
                   'Value': [average_discount_rate, average_quantity_per_order, average_sales_per_order, average_profit_per_order]}

kpi_df_1 = pd.DataFrame(KPIs_1)


# In[23]:


# Total Sales and Profit by Sub-Category
subcategory_sales_profit = df.groupby('Product Sub-Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

# Sort the DataFrame by Sales in descending order
subcategory_sales_profit_sorted = subcategory_sales_profit.sort_values(by='Sales', ascending=False)

# Plotting Sales and Profit by Sub-Category in descending order
plt.figure(figsize=(15, 7))

# Plotting Sales
sns.barplot(x='Product Sub-Category', y='Sales', data=subcategory_sales_profit_sorted, color='skyblue', label='Sales')

# Plotting Profit
sns.barplot(x='Product Sub-Category', y='Profit', data=subcategory_sales_profit_sorted, color='orange', label='Profit')

plt.title('Total Sales and Profit by Product Sub-Category (Descending Order)')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[24]:


# Total Quantity and Number of Orders by Product Category
category_data = df.groupby('Product Category').agg({'Quantity': 'sum', 'Order ID': 'nunique'}).reset_index()

# Sorting the DataFrame by Quantity in descending order
category_data_sorted = category_data.sort_values(by='Quantity', ascending=False)

# Melting the DataFrame for easier plotting
category_data_melted = category_data_sorted.melt(id_vars='Product Category', var_name='Metric', value_name='Total')

# Grouped bar chart for Quantity and Number of Orders by Product Category
plt.figure(figsize=(15, 7))
sns.barplot(data=category_data_melted, x='Product Category', y='Total', hue='Metric')
plt.title('Total Quantity and Number of Orders by Product Category (Descending Order)')
plt.xlabel('Product Category')
plt.ylabel('Total')
plt.xticks(rotation=45)
plt.show()


# In[25]:


# Create a clustered bar chart for average Unit Cost Price, Unit Profit, and Unit Discounted Sale Price by Product Category
plt.figure(figsize=(14, 8))
sns.barplot(x='Product Category', y='value', hue='variable', data=pd.melt(df, id_vars='Product Category', value_vars=['Unit Cost Price', 'Unit Profit', 'Unit Discounted Sale Price']))
plt.title('Unit Cost Price, Unit Profit, and Unit Discounted Sale Price by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Average Value')
plt.legend(title='Variable', title_fontsize='12')
plt.xticks(rotation=0)
plt.show()



# In[26]:


# Total Quantity and Number of Orders by Product Sub-Category
subcategory_data = df.groupby('Product Sub-Category').agg({'Quantity': 'sum', 'Order ID': 'nunique'}).reset_index()

# Sorting the DataFrame by Quantity in descending order
subcategory_data_sorted = subcategory_data.sort_values(by='Quantity', ascending=False)

# Melting the DataFrame for easier plotting
subcategory_data_melted = subcategory_data_sorted.melt(id_vars='Product Sub-Category', var_name='Metric', value_name='Total')

# Grouped bar chart for Quantity and Number of Orders by Product Sub-Category
plt.figure(figsize=(15, 7))
sns.barplot(data=subcategory_data_melted, x='Product Sub-Category', y='Total', hue='Metric')
plt.title('Total Quantity and Number of Orders by Product Sub-Category (Descending Order)')
plt.xlabel('Product Sub-Category')
plt.ylabel('Total')
plt.xticks(rotation=45)
plt.show()


# In[27]:


print(df.columns)



# In[28]:


# Total Profit Margin by Product Category
category_profit_margin = df.groupby('Product Category').agg({'Profit Margin': 'mean'}).reset_index()

# Sort the DataFrame by Profit Margin in descending order
category_profit_margin = category_profit_margin.sort_values(by='Profit Margin', ascending=False)

# Plotting Profit Margin by Product Category
plt.figure(figsize=(15, 8 ))
bar_plot = sns.barplot(x='Product Category', y='Profit Margin', data=category_profit_margin)
plt.ylabel('Profit Margin (%)')  # Changing y-axis title here
plt.title('Average Profit Margin by Product Category')
plt.xticks(rotation=0)

# Annotate the bars with profit margin values
for product_category, profit_margin in enumerate(category_profit_margin['Profit Margin']):
    bar_plot.text(product_category, profit_margin , f'{profit_margin:.2f}%', ha='center', va='bottom', fontsize=10, color='black')

plt.show()

# Total Profit Margin by Product Sub-Category
subcategory_profit_margin = df.groupby('Product Sub-Category').agg({'Profit Margin': 'mean'}).reset_index()

# Sort the DataFrame by Profit Margin in descending order
subcategory_profit_margin = subcategory_profit_margin.sort_values(by='Profit Margin', ascending=False)

# Plotting Profit Margin by Product Sub-Category
plt.figure(figsize=(15, 7))
bar_plot = sns.barplot(x='Product Sub-Category', y='Profit Margin', data=subcategory_profit_margin)
plt.ylabel('Profit Margin (%)')  # Changing y-axis title here
plt.title('Average Profit Margin by Product Sub-Category')
plt.xticks(rotation=45, ha='right')

plt.show()


# In[29]:


# Top 10 products by sales
top_products_sales = df.groupby('Product Name')['Sales'].sum().reset_index()
top_products_sales = top_products_sales.sort_values(by='Sales', ascending=False).head(10)

# Creating a bar chart for the top 10 products based on sales
plt.figure(figsize=(15, 7))
sns.barplot(x='Sales', y='Product Name', data=top_products_sales, color='lightblue')
plt.title('Top 10 Products Based on Sales')
plt.ylabel('Product Name')
plt.xlabel('Total Sales')
plt.xticks(rotation=90, ha='right')  # Rotate product names for better readability
plt.show()

# Top 10 products by profits
top_products_profit = df.groupby('Product Name')['Profit'].sum().reset_index()
top_products_profit = top_products_profit.sort_values(by='Profit', ascending=False).head(10)

# Creating a bar chart for the top 10 products based on profit
plt.figure(figsize=(15, 7))
sns.barplot(x='Profit', y='Product Name', data=top_products_profit, color='lightblue')
plt.title('Top 10 Products Based on Profit')
plt.ylabel('Product Name')
plt.xlabel('Total Profit')
plt.xticks(rotation=90, ha='right')  # Rotate product names for better readability
plt.show()


# In[30]:


# Bottom 10 products by sales
bottom_products_sales = df.groupby('Product Name')['Sales'].sum().reset_index()
bottom_products_sales = bottom_products_sales.sort_values(by='Sales', ascending=True).head(10)

# Creating a bar chart for the bottom 10 products based on sales
plt.figure(figsize=(15, 5))
sns.barplot(x='Sales', y='Product Name', data=bottom_products_sales, color='lightcoral')
plt.title('10 Products with least Sales')
plt.ylabel('Product Name')
plt.xlabel('Total Sales')
plt.xticks(rotation=0, ha='right')  # Rotate product names for better readability
plt.show()

# Bottom 10 products by Profit
bottom_products_profit = df.groupby('Product Name')['Profit'].sum().reset_index()
bottom_products_profit = bottom_products_profit.sort_values(by='Profit', ascending=True).head(10)

# Creating a bar chart for the bottom 10 products based on sales
plt.figure(figsize=(15, 7))
sns.barplot(x='Profit', y='Product Name', data=bottom_products_profit, color='lightcoral')
plt.title('10 loss making products ')
plt.ylabel('Product Name')
plt.xlabel('Profit')
plt.xticks(rotation=90, ha='right')  # Rotate product names for better readability
plt.show()


# In[31]:


# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Set 'Order Date' as the index
df.set_index('Order Date', inplace=True)

# Resample the data on a monthly frequency and sum the 'Sales'
monthly_sales = df['Sales'].resample('M').sum()

# Create a line chart for Monthly Sales Trends using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, lw=2, label='Monthly Sales')

plt.title('Monthly Sales Trends Analysis')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.legend()
plt.show()


# In[32]:


# Resampling the data on a monthly frequency and sum the 'Sales'
monthly_sales = df['Sales'].resample('M').sum()

# Identifying the highest and lowest sales values and their respective dates
max_sales_date = monthly_sales.idxmax()
max_sales_value = monthly_sales.max()
min_sales_date = monthly_sales.idxmin()
min_sales_value = monthly_sales.min()

# Printing the highest and lowest sales values and their dates
print(f'Highest Sales: {max_sales_value:,.2f} on {max_sales_date.strftime("%b %Y")}')
print(f'Lowest Sales: {min_sales_value:,.2f} on {min_sales_date.strftime("%b %Y")}')


# In[33]:


# Resample the data on a monthly frequency and sum the 'Profit'
monthly_profit = df['Profit'].resample('M').sum()

# Create an area chart for Monthly Profit Trends using Matplotlib
plt.figure(figsize=(10, 6))
plt.fill_between(monthly_profit.index, monthly_profit.values, color='green', alpha=0.3, label='Monthly Profit')

plt.title('Monthly Profit Trends Analysis')
plt.xlabel('Month')
plt.ylabel('Total Profit')
plt.legend()
plt.show()


# In[34]:


# Calculate profit margin
df['Profit Margin'] = (df['Profit'] / df['Sales']) * 100

# Create a histogram
plt.figure(figsize=(12, 8))
plt.hist(df['Profit Margin'], bins=100, edgecolor='black', alpha=0.7)
plt.axvline(df['Profit Margin'].median(), color='red', linestyle='dashed', linewidth=2, label='Median Profit Margin')
plt.title('Histogram of Profit Margin')
plt.xlabel('Profit Margin (%)')
plt.ylabel('Number of Orders')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# Show the plot
plt.show()


# In[35]:


# Total Discounted Sales, Profits, and Sales without Discount
discounted_data = df[df['Discount'] > 0]
no_discount_data = df[df['Discount'] == 0]

discounted_sales_total = discounted_data['Sales'].sum()
no_discount_sales_total = no_discount_data['Sales'].sum()

discounted_profit_total = discounted_data['Profit'].sum()
no_discount_profit_total = no_discount_data['Profit'].sum()

# Creating a DataFrame for the totals
totals_data = pd.DataFrame({
    'Sales Type': ['Discounted', 'No Discount'],
    'Total Sales': [discounted_sales_total, no_discount_sales_total],
    'Total Profit': [discounted_profit_total, no_discount_profit_total]
})

# Grouped bar chart for Discounted Sales and Sales without Discount
plt.figure(figsize=(10, 8))
sns.barplot(data=totals_data, x='Sales Type', y='Total Sales', palette='viridis', label='Sales')
sns.barplot(data=totals_data, x='Sales Type', y='Total Profit', palette='magma', label='Profit', alpha=0.7)
plt.title('Total Discounted Sales, Profits, and Sales without Discount')
plt.xlabel('Sales Type')
plt.ylabel('Total')
plt.legend()
plt.show()


# In[36]:


# Selecting the columns of interest for correlation
data_for_correlation = df[['Discount', 'Sales','Profit']]

# Calculating the correlation matrix
correlation_matrix = data_for_correlation.corr()

# Creating a heatmap for visualization
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True , cmap='coolwarm')
plt.title('Correlation Matrix for Discount, Sales, and Profit')
plt.show()


# In[37]:


# Find the maximum profit
max_profit = monthly_profit.max()
max_profit_date = monthly_profit.idxmax()

# Find the minimum profit
min_profit = monthly_profit.min()
min_profit_date = monthly_profit.idxmin()

print(f'Maximum Profit: {max_profit:,.2f}, Date: {max_profit_date.strftime("%b %Y")}')
print(f'Minimum Profit: {min_profit:,.2f}, Date: {min_profit_date.strftime("%b %Y")}')


# In[38]:


# Get the order counts for each shipping mode
order_counts = df['Ship Mode'].value_counts()

# Create a bar plot for number of orders based on shipping mode
plt.figure(figsize=(12, 8))
sns.barplot(x=order_counts.index, y=order_counts.values, order=order_counts.index, palette='viridis')
plt.title('Number of Orders Based on Shipping Mode')
plt.xlabel('Shipping Mode')
plt.ylabel('Number of Orders')
plt.show()


# In[39]:


# Total Quantity and Number of Orders by Product Sub-Category
Region_data = df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

# Sorting the DataFrame by Quantity in descending order
Region_data_sorted = Region_data.sort_values(by='Sales', ascending=False)

# Melting the DataFrame for easier plotting
Region_data_melted = Region_data_sorted.melt(id_vars='Region', var_name='Metric', value_name='Total')

# Grouped bar chart for Quantity and Number of Orders by Product Sub-Category
plt.figure(figsize=(15, 7))
sns.barplot(data=Region_data_melted, x='Region', y='Total', hue='Metric')
plt.title('Total Sales and Profit by Region')
plt.xlabel('Region')
plt.ylabel('Total')
plt.xticks(rotation=45)
plt.show()


# In[40]:


# Calculate profitability for each region
Region_data['Profitability'] = (Region_data['Profit'] / Region_data['Sales']) * 100

# Print the DataFrame with profitability values
print(Region_data[['Region', 'Profitability','Sales','Profit']])


# In[41]:


# Top 10 states by sales
# Grouping the data by 'State' and calculate total sales
state_wise_sales = df.groupby('State')['Sales'].sum().reset_index()

# Sorting the DataFrame by 'Sales' in descending order and take the top 10 states
top_10_states_sales = state_wise_sales.sort_values(by='Sales', ascending=False).head(10)

# Creating a bar chart for the top 10 states based on sales
plt.figure(figsize=(15, 7))
sns.barplot(x='Sales', y='State', data=top_10_states_sales, color='lightblue')
plt.title('Top 10 States Based on Sales')
plt.ylabel('Total Sales')
plt.xlabel('State')
plt.xticks(rotation=45, ha='right')  # Rotate state names for better readability
plt.show()


# In[42]:


# Calculate the total sales for the entire dataset
total_sales_all_states = df['Sales'].sum()

# Get the top 2 states with the highest sales
total_sales_top_2_states = top_10_states_sales.head(2)['Sales'].sum()

# Count the total number of unique states
total_states_count = df['State'].nunique()

# Get the names of the top 2 states with the highest sales
top_2_state_names = top_10_states_sales.head(2)['State']

# Print the names of the top 2 states
print("Top 2 States:")
for state_name in top_2_state_names:
    print(state_name)

# Calculate the sales share for each of the top 2 states
sales_share_top_2_states = (total_sales_top_2_states / total_sales_all_states) * 100

# Print the sales share
print(f'Sales Share of Top 2 States: {sales_share_top_2_states:.2f}%')

# Print the total number of states
print(f'Total Number of States: {total_states_count}')


# In[43]:


# Top 10 states based on profit
# Grouping the data by 'State' and calculate total profit
state_wise_profit = df.groupby('State')['Profit'].sum().reset_index()

# Sorting the DataFrame by 'Profit' in descending order and take the top 10 states
top_10_states_profit = state_wise_profit.sort_values(by='Profit', ascending=False).head(10)

# Creating a bar chart for the top 10 states based on profit
plt.figure(figsize=(15, 7))
sns.barplot(x='Profit', y='State', data=top_10_states_profit, color='orange')
plt.title('Top 10 States Based on Profit')
plt.ylabel('Total Profit')
plt.xlabel('State')
plt.xticks(rotation=45, ha='right')  # Rotate state names for better readability
plt.show()


# In[44]:


# Calculate the total profit for the entire dataset
total_profit_all_states = df['Profit'].sum()

# Get the top 2 states with the highest profit
total_profit_top_2_states = top_10_states_profit.head(2)['Profit'].sum()

# Calculate the profit share for each of the top 2 states
profit_share_top_2_states = (total_profit_top_2_states / total_profit_all_states) * 100

# Print the profit share
print(f'Profit Share of Top 2 States: {profit_share_top_2_states:.2f}%')


# In[45]:


# Filter the DataFrame to get data for 'New York' and 'California'
ny_data = df[df['State'] == 'New York']
ca_data = df[df['State'] == 'California']

# Calculate the total sales and profit for 'New York'
ny_sales = ny_data['Sales'].sum()
ny_profit = ny_data['Profit'].sum()

# Calculate the total sales and profit for 'California'
ca_sales = ca_data['Sales'].sum()
ca_profit = ca_data['Profit'].sum()

# Print the values for 'New York' and 'California'
print("State: New York, Sales:", ny_sales, "Profit:", ny_profit)
print("State: California, Sales:", ca_sales, "Profit:", ca_profit)

# Calculate the total sales and profit for the entire dataset
total_sales_all_states = df['Sales'].sum()
total_profit_all_states = df['Profit'].sum()

# Print the total sales and profit
print(f'Total Sales for All States: ${total_sales_all_states:.2f}')
print(f'Total Profit for All States: ${total_profit_all_states:.2f}')


# In[46]:


# Grouping the data by 'State' and calculate total sales
state_wise_sales = df.groupby('State')['Sales'].sum().reset_index()

# Sorting the DataFrame by 'Sales' in ascending order and take the bottom 10 states
lowest_10_states_sales = state_wise_sales.sort_values(by='Sales', ascending=True).head(10)

# Creating a bar chart for the states with the lowest 10 sales
plt.figure(figsize=(15, 7))
sns.barplot(x='Sales', y='State', data=lowest_10_states_sales, color='lightcoral')
plt.title('States with the Lowest Sales')
plt.ylabel('Total Sales')
plt.xlabel('State')
plt.xticks(rotation=45, ha='right')  # Rotate state names for better readability
plt.show()


# In[47]:


# Grouping the data by 'State' and calculate total profit
state_wise_profit = df.groupby('State')['Profit'].sum().reset_index()

# Sorting the DataFrame by 'Profit' in ascending order and take the bottom 10 states
lowest_10_states_profit = state_wise_profit.sort_values(by='Profit', ascending=True).head(10)

# Creating a bar chart for the states with the lowest 10 profit
plt.figure(figsize=(15, 7))
sns.barplot(x='Profit', y='State', data=lowest_10_states_profit, color='lightcoral')
plt.title('Loss making states')
plt.ylabel('Total Profit')
plt.xlabel('State')
plt.xticks(rotation=45, ha='right')  # Rotating state names for better readability
plt.show()


# In[48]:


# Melting the DataFrame to create 'Metric' and 'Amount' columns
segment_summary_melted = pd.melt(df, id_vars='Customer Segment', value_vars=['Sales', 'Profit'], var_name='Metric', value_name='Amount')

# Sorting the segments by total sales in descending order
sorted_segments = segment_summary_melted[segment_summary_melted['Metric'] == 'Sales'].sort_values(by='Amount', ascending=False)['Customer Segment'].unique()

# Plotting grouped bar chart for Sales and Profit based on sorted customer segments
plt.figure(figsize=(15, 7))
ax = sns.barplot(x='Customer Segment', y='Amount', hue='Metric', data=segment_summary_melted, palette='viridis', order=sorted_segments)
plt.title('Customer Segments Based on Sales and Profit (Sorted by Sales)')
plt.ylabel('Total Amount')

plt.show()


# In[49]:


# Group data by 'Customer Segment' and calculate total number of orders and total quantity
segment_orders_quantity = df.groupby('Customer Segment').agg({'Order ID': 'nunique', 'Quantity': 'sum'}).reset_index()

# Sorting the DataFrame by number of orders in descending order
segment_orders_quantity_sorted = segment_orders_quantity.sort_values(by='Order ID', ascending=False)

# Melting the DataFrame for easier plotting
segment_orders_quantity_melted = segment_orders_quantity_sorted.melt(id_vars='Customer Segment', var_name='Metric', value_name='Total')

# Grouped bar chart for Orders and Quantity by Customer Segment
plt.figure(figsize=(15, 7))
sns.barplot(data=segment_orders_quantity_melted, x='Customer Segment', y='Total', hue='Metric')

plt.title('Total Orders and Quantity by Customer Segment (Descending Order)')
plt.xlabel('Customer Segment')
plt.ylabel('Total')
plt.xticks(rotation=45)
plt.show()


# In[50]:


#Customers with most orders
# Grouping data by 'Customer Name' and calculate total number of orders
customer_orders = df.groupby('Customer Name')['Order ID'].nunique().reset_index()

# Sorting the DataFrame by number of orders in descending order
customer_orders = customer_orders.sort_values(by='Order ID', ascending=False)

# Calculating mean number of orders
mean_orders = customer_orders['Order ID'].mean()

# Plotting bar chart for number of orders based on customers
plt.figure(figsize=(15, 7))
ax = sns.barplot(x='Customer Name', y='Order ID', data=customer_orders.head(27), palette='viridis')


# Adding a line for mean number of orders
plt.axhline(y=mean_orders, color='red', linestyle='--', label='Mean Orders')

plt.title('Customers with Most Orders')
plt.ylabel('Number of Orders')
plt.xlabel('Customer Name')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()


# Filtering customers with orders above the mean value
customers_above_mean = customer_orders[customer_orders['Order ID'] > mean_orders]

# Displaying total number of customers above the mean
total_customers_above_mean = len(customers_above_mean)
print(f'Total customers with orders above the mean ({mean_orders}): {total_customers_above_mean}')


# In[51]:


# Sorting the DataFrame by number of orders in ascending order
customer_orders1 = customer_orders.sort_values(by='Order ID', ascending=True)

# Plotting bar chart for number of orders based on customers
plt.figure(figsize=(15, 7))
ax = sns.barplot(x='Customer Name', y='Order ID', data=customer_orders1.head(40), palette='viridis')

# Adding a line for mean number of orders
plt.axhline(y=mean_orders, color='red', linestyle='--', label='Mean Orders')

plt.title('Customers with Fewest Orders')
plt.ylabel('Number of Orders')
plt.xlabel('Customer Name')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

# Filtering customers with orders below the mean value
customers_below_mean = customer_orders[customer_orders['Order ID'] < mean_orders]

# Displaying total number of customers below the mean
total_customers_below_mean = len(customers_below_mean)
print(f'Total customers with orders below the mean ({mean_orders}): {total_customers_below_mean}')

# Displaying the list of customer names below the median in tabular format
table_data = customers_below_mean[['Customer Name']]
table_data.columns = ['Customer Name']

print(f'\nList of customers below the mean:\n{table_data.to_string(index=False)}')


# In[52]:


# Grouping data by 'Customer Name' and calculate total quantity of items ordered
customer_quantity = df.groupby('Customer Name')['Quantity'].sum().reset_index()

# Sorting the DataFrame by total quantity in descending order
customer_quantity = customer_quantity.sort_values(by='Quantity', ascending=False)

# Calculating mean quantity of items
mean_quantity = customer_quantity['Quantity'].mean()

# Plotting bar chart for total quantity of items based on customers
plt.figure(figsize=(15, 7))
ax = sns.barplot(x='Customer Name', y='Quantity', data=customer_quantity.head(27), palette='viridis')

# Adding a line for mean quantity of items
plt.axhline(y=mean_quantity, color='red', linestyle='--', label='Mean Quantity')

plt.title('Customers with Most Orders')
plt.ylabel('Total Quantity of Items')
plt.xlabel('Customer Name')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

# Filtering customers with total quantity above the mean value
customers_above_mean_quantity = customer_quantity[customer_quantity['Quantity'] > mean_quantity]

# Displaying total number of customers above the mean quantity
total_customers_above_mean_quantity = len(customers_above_mean_quantity)
print(f'Total customers with quantity above the mean ({mean_quantity}): {total_customers_above_mean_quantity}')
print('Mean Quantity-' + str(mean_quantity))


# In[53]:


# Sorting the DataFrame by total quantity in descending order
customer_quantity = customer_quantity.sort_values(by='Quantity', ascending=True)

# Plotting bar chart for total quantity of items based on customers
plt.figure(figsize=(15, 7))
ax = sns.barplot(x='Customer Name', y='Quantity', data=customer_quantity.head(27), palette='viridis')

# Adding a line for mean quantity of items
plt.axhline(y=mean_quantity, color='red', linestyle='--', label='Mean Quantity')

plt.title('Customers with Least Orders')
plt.ylabel('Total Quantity of Items')
plt.xlabel('Customer Name')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

# Filtering customers with total quantity above the mean value
customers_above_mean_quantity = customer_quantity[customer_quantity['Quantity'] < mean_quantity]

# Displaying total number of customers above the mean quantity
total_customers_above_mean_quantity = len(customers_above_mean_quantity)
print(f'Total customers with quantity above the mean ({mean_quantity}): {total_customers_above_mean_quantity}')


# In[54]:


# Calculate correlation coefficient between Profit and Discount
correlation_profit_discount = df['Profit'].corr(df['Discount'])

# Calculate correlation coefficient between Sales and Discount
correlation_sales_discount = df['Sales'].corr(df['Discount'])

print(f'Correlation between Profit and Discount: {correlation_profit_discount:.2f}')
print(f'Correlation between Sales and Discount: {correlation_sales_discount:.2f}')


# In[55]:


# Calculate total sales with discount and without discount
total_sales_with_discount = df[df['Discount'] > 0]['Sales'].sum()
total_sales_without_discount = df[df['Discount'] == 0]['Sales'].sum()

# Create a pie chart
labels = ['Discounted Sales', 'Sales without Discount']
sizes = [total_sales_with_discount, total_sales_without_discount]
colors = ['skyblue', 'lightcoral']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.4))

# Adding a title
plt.title('Comparison of Discounted Sales and Sales without Discount')

plt.show()


# In[56]:


print(df.columns)


# In[57]:


column_names = df.columns.tolist()
print(column_names)


# In[ ]:




