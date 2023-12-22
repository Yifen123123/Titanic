import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('test.csv')
test_data = pd.read_csv('test.csv')

# 假設你有一個包含性別、艙等和年齡的 DataFrame
# 如果你的 DataFrame 不同，請根據實際情況調整代碼

# 使用 Seaborn 繪製散點圖
sns.scatterplot(data=test_data, x='Pclass', y='Age', hue='Sex', size='Age', sizes=(20, 200))

# 添加標題和軸標籤
plt.title('Relationship between Pclass, Age, and Sex')
plt.xlabel('Pclass')
plt.ylabel('Age')

# 顯示圖形
plt.show()