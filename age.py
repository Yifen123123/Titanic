import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 文件
data = pd.read_csv('test.csv')
# 文件中'Age'列
age_data = data['Age']

# 繪製直方圖
plt.hist(age_data, bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Passenger Ages')
plt.show()