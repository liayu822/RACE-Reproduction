import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 30})  

data = np.array([
    [12, 46, 70, 74, 76],
    [15, 58, 78, 86, 92],
    [15, 70, 84, 94, 96]
])


x_labels = ['1', '2', '3', '4', '5']  
y_labels = ['Gemma', 'Qwen', 'GLM']   

plt.figure(figsize=(10, 6))


sns.heatmap(data, 
            annot=True,  
            fmt='.1f',   
            cmap='YlOrRd', 
            xticklabels=x_labels,
            yticklabels=y_labels,
            cbar=True)   

plt.tight_layout()

plt.savefig('./man_heatmap.pdf', format='pdf', bbox_inches='tight')

plt.show()
