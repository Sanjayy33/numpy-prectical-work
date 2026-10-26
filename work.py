import numpy as np
np.random.seed(50)

# Section 1 :

# Q.1 Write NumPy code to create a random integer array of shape (30, 5) with values between 50 and 500. Name it sales_data.
sales_data = np.random.randint(low=50,high=500,size=150)
reshape_slaes_data = sales_data.reshape(30,5)
print(reshape_slaes_data)

# Q.2 What function would you use to check the shape, data type, and number of dimensions of sales_data?
shape_of_data = reshape_slaes_data.shape
dtpye_of_data = sales_data.dtype
dimensions_of_data = sales_data.ndim
print(shape_of_data)
print(dtpye_of_data)
print(dimensions_of_data)

# Q.3 How many total sales values are stored in this array? Write the NumPy expression to find this.
total_value_In_data = sales_data.size
print(total_value_In_data)

# Q.4 Extract the sales data for just the first product (column 0). What is the resulting shape?
extract_sales_data = reshape_slaes_data[:,:1]
print(reshape_slaes_data)
print(extract_sales_data)

# Q.5 Slice the data to get sales for days 10 to 20 (inclusive) for all products.
get_10_to_20_days_data = reshape_slaes_data[10:20,::]
print(get_10_to_20_days_data)



# Section 2:

# Q.6 Calculate the total sales for each product across all 30 days.
total_sales = sales_data.sum()
print(total_sales)

# Q.7 Find the average daily sales across all products combined.
avg_sales = sales_data.mean()
print(avg_sales)

# Q.8 Which product had the highest total sales? Write the NumPy code to find its index.
highest_total_sales_index = sales_data.argmax()
print(highest_total_sales_index)

# Q.9 Find the standard deviation of sales for each product. What does a high std dev indicate?
std_sales = sales_data.std()
print(std_sales)

# Q.10  Calculate the median daily sales for the entire dataset.
median_sales = np.median(sales_data)
print(median_sales)

# Q.11 Find the minimum and maximum sales values recorded in the entire dataset.
max_total_sales = sales_data.max()
min_total_sales = sales_data.min()
print(max_total_sales)
print(min_total_sales)



# Section 3:

# Q.12 Write code to find all days where sales of Product 3 exceeded 300 units.
product_3 = reshape_slaes_data[::,2:3]
product_3_sales = []
print(product_3_sales)

# Q.13 How many days had total sales (across all products) greater than 1500?
daily_total = reshape_slaes_data.sum(axis=1)
count = np.sum(daily_total > 1500)  
print(daily_total)
print(daily_total > 1500)
print(count)

# Q.14 Replace all sales values below 60 with 60 (treat these as the minimum threshold). Write the NumPy code.
sales_copy = reshape_slaes_data.copy()
sales_copy[sales_copy < 60] = 60
print(sales_copy)

# Q.15 Find the indices (day numbers) where Product 1 had its top 5 highest sales days.
product1 = reshape_slaes_data[:, 0]
top5_idx = np.argsort(product1)[-5:][::-1]
print(product1)
print(top5_idx)
print(top5_idx + 1)
print(product1[top5_idx])



# Section 4:

# Q.16  Each product has a different profit margin: [0.2, 0.35, 0.15, 0.4, 0.25]. Multiply the sales array by this margin array to get profit data. What NumPy feature makes this possible?
profile_margin = np.array([0.2, 0.35, 0.15, 0.4, 0.25])
sales_multiply = reshape_slaes_data * profile_margin
print(sales_multiply)

# Q.17 Normalise the sales data for each product so that values range between 0 and 1.
sales_min = sales_data.min()
sales_max = sales_data.max()
normalise_sales = (sales_data - sales_min) / (sales_max - sales_min)
print(normalise_sales)

# Q.18 Compute the percentage change in total daily sales from one day to the next.
daily_total = reshape_slaes_data.sum(axis=1)
pct_change = np.diff(daily_total) / daily_total[:-1]*100
print(pct_change)

# Q.19 Stack a new row representing forecast sales for Day 31: [200, 310, 150, 420, 280]. What function would you use?
day_31 = [200,310,150,420,280]
update_sales = np.vstack((reshape_slaes_data, day_31))
print(update_sales)

# Q.20 Reshape the entire sales_data array into a 1D array. How many elements does it contain?
reshape_1D = reshape_slaes_data.flatten()
print(reshape_1D)



# Section 5:

# Q.21 Which 3 days had the highest combined sales across all products? Write code to find them.
top3_idx = np.argsort(daily_total)[-3:][::-1]
print(top3_idx)

# Q.22 Compute a 7-day rolling average for Product 2's sales using NumPy. (No pandas allowed!)
product2 = reshape_slaes_data[:, 1]      # Product 2 = column index 1
window = 7
rolling_avg = []
for i in range(len(product2) - window + 1):
    window_data = product2[i : i + window]
    rolling_avg.append(np.mean(window_data))
rolling_avg = np.array(rolling_avg)
print(rolling_avg)

# Q.23 The company wants to give a 10% bonus for every day a product exceeds 400 units. Calculate total bonus units per product.
mask = sales_data > 400
bonus = np.where(mask, sales_data * 0.10, 0)
total_bonus = bonus.sum(axis=0)
print(total_bonus)

# Q.24 Compare the first 15 days vs last 15 days: which half had better average sales per product?
half = len(sales_data) // 2
first_avg = sales_data[:half].mean(axis=0)
last_avg  = sales_data[-half:].mean(axis=0)
better = np.where(last_avg > first_avg, "Last half", "First half")
pct_change = (last_avg - first_avg) / first_avg * 100
print(first_avg)
print(last_avg)
print(better)
print(pct_change.round(2))

# Q.25 Save the final cleaned array to a file called cleaned_sales.npy and write code to reload it.
np.save("cleaned_sales.npy", sales_data)
loaded_sales = np.load("cleaned_sales.npy")
print(loaded_sales)
print("Same data?", np.array_equal(sales_data, loaded_sales))
print("Shape:", loaded_sales.shape)
print("dtype:", loaded_sales.dtype)