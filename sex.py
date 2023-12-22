import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 文件
data = pd.read_csv('test.csv')

# 繪製性別分佈的長條圖
gender_counts = data['Sex'].value_counts()
print(gender_counts)
print(type(gender_counts))
gender_counts.plot(kind='bar', color=['blue', 'pink'])
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Distribution of Passengers by Gender')
plt.show()