# Ex.No : 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
## DATE : 18/03/2025
## AIM :
To Implement Linear and Polynomial Trend Estiamtion Using Python.

## ALGORITHM :
1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using least square method
4. Calculate the polynomial trend values using least square method
5. End the program
   
## PROGRAM :
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path = "Gold Price Prediction.csv"
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
resampled_data = data['Price Today'].resample('Y').mean().to_frame()
resampled_data.index = resampled_data.index.year
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Date': 'Year'}, inplace=True)
years = resampled_data['Year'].tolist()
prices = resampled_data['Price Today'].tolist()
X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, prices)]
```
#### A - LINEAR TREND ESTIMATION :
```
n = len(years)
b = (n * sum(xy) - sum(prices) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(prices) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]
```
#### B- POLYNOMIAL TREND ESTIMATION  
```
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, prices)]
coeff = [[n, sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]
Y = [sum(prices), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]
```
#### STORE TRENDS IN DATAFRAME :
```
resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend
resampled_data.set_index('Year', inplace=True)
```

#### PLOT RESULTS : 
```
plt.figure(figsize=(10, 5))
plt.plot(resampled_data.index, resampled_data['Price Today'], linestyle='--', color='blue', label='Gold Price')
plt.plot(resampled_data.index, resampled_data['Linear Trend'], marker='o', linestyle='-', color='black', label='Linear Trend')
plt.xlabel('Year')
plt.ylabel('Gold Price')
plt.title('Gold Price Trends')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(resampled_data.index, resampled_data['Price Today'], linestyle='--', color='black', label='Gold Price')
plt.plot(resampled_data.index, resampled_data['Polynomial Trend'], marker='o', linestyle='-', color='pink', label='Polynomial Trend')
plt.xlabel('Year')
plt.ylabel('Gold Price')
plt.title('Gold Price with Polynomial Trend')
plt.legend()
plt.grid(True)
plt.show()
print(f"Linear Trend: y = {a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y = {a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")
```

## OUTPUT :
#### A - LINEAR TREND ESTIMATION :
![image](https://github.com/user-attachments/assets/334379f7-7bf4-4926-86a4-74672f61e181)

#### B- POLYNOMIAL TREND ESTIMATION :
![image](https://github.com/user-attachments/assets/10835c0b-2356-4d92-a924-7e576f49bfac)


## RESULT :
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
