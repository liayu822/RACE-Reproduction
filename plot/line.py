import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 30})  

x = np.arange(1, 6)  
y1 = [14, 56, 84, 82, 88][:5]  
y2 = [20, 68, 96, 90, 96][:5]  
y3 = [14, 88, 100, 100, 98][:5]  

plt.figure(figsize=(10, 6))

plt.plot(x, y1, color='#1f77b4', marker='o', label='Gemma')
plt.plot(x, y2, color='#ff7f0e', marker='s', label='Qwen')
plt.plot(x, y3, color='lightcoral', marker='D', label='GLM')

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xlabel('Number of turns')
plt.ylabel('ASR (%)')

plt.xticks(x) 


plt.savefig('./turn1.pdf', format='pdf', bbox_inches='tight')


plt.show()
