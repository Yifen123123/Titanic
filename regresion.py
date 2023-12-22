import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('test.csv')
test_data = pd.read_csv('test.csv')


# 使用 Seaborn 繪製數據點和回歸直線
sns.regplot(data=test_data, x='Pclass', y='Age')

# 添加標題和軸標籤
plt.title('Regression Plot of Age by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Age')

# 顯示圖形
plt.show()