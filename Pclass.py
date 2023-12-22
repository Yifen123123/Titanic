import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('test.csv')

# 繪製艙等分佈的長條圖
pclass_counts = data['Pclass'].value_counts().sort_index()
pclass_counts.plot(kind='bar', color=['green', 'orange', 'blue'])
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.title('Distribution of Passengers by Pclass')
plt.show()