import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 30}) 

models = ['Gemma', 'Qwen', 'GLM']  
success_rates_70b = [84, 96, 100]  
success_rates_7b = [74,84,86]   
success_rates_additional_1 = [78,90,92]  
success_rates_additional_2 = [72,86,92]  

fig, ax = plt.subplots(figsize=(10, 7))

bar_width = 0.2
index = np.arange(len(models))


bars_70b = ax.bar(index, success_rates_70b, bar_width, label='RACE', color='#1f77b4', hatch='/')
bars_7b = ax.bar(index + bar_width, success_rates_7b, bar_width, label='w/o GE', color='#ff7f0e', hatch='\\')
bars_additional_1 = ax.bar(index + 2*bar_width, success_rates_additional_1, bar_width, label='w/o SP', color='#2ca02c', hatch='x')
bars_additional_2 = ax.bar(index + 3*bar_width, success_rates_additional_2, bar_width, label='w/o RF', color='lightcoral', hatch='+')

def add_labels(bars, data):
    for bar, value in zip(bars, data):
        yval = bar.get_height() 
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 2, str(value), ha='center', va='bottom', color='black')

add_labels(bars_70b, success_rates_70b)
add_labels(bars_7b, success_rates_7b)
add_labels(bars_additional_1, success_rates_additional_1)
add_labels(bars_additional_2, success_rates_additional_2)


ax.set_ylabel('ASR (%)')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(models)
ax.legend()

plt.show()

plt.savefig('./ab.pdf', format='pdf', bbox_inches='tight')
