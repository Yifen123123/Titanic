import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
test_data = pd.read_csv('test.csv')
# 假設你有一個包含性別、艙等、年齡和存活狀態的 DataFrame
# 這裡使用的是你之前提到的 train_data
# 如果你的 DataFrame 不同，請根據實際情況調整代碼
# 使用 Seaborn 繪製散點圖
sns.scatterplot(data=test_data, x='Pclass', y='Age', hue='Survived', size='Age', sizes=(20, 200))

# 添加標題和軸標籤
plt.title('Relationship between Pclass, Age, and Survived')
plt.xlabel('Pclass')
plt.ylabel('Age')

# 顯示圖形
plt.show()

# 使用 Seaborn 繪製箱形圖
sns.boxplot(x='Pclass', y='Age', hue='Survived', data=train_data)

# 添加標題和軸標籤
plt.title('Boxplot of Age by Pclass and Survived')
plt.xlabel('Pclass')
plt.ylabel('Age')

# 顯示圖形
plt.show()