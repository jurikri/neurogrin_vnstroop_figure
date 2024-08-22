#%%

import numpy as np
import matplotlib.pyplot as plt
import pickle

if False:
    msdict_figure4 = {
    'epochs': epochs,
    'mean_train_loss': mean_train_loss,
    'sem_train_loss': sem_train_loss,
    'scatter_x': scatter_x,
    'scatter_y': scatter_y}

spath = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure' + '\\'

if False:
    with open(spath + 'vnstroop_figure4.pickle', 'wb') as file:
        pickle.dump(msdict, file)
    
with open(spath + 'vnstroop_figure4.pickle', 'rb') as file:
    msdict = pickle.load(file)

epochs = msdict['epochs']
mean_train_loss = msdict['mean_train_loss']
sem_train_loss = msdict['sem_train_loss']
scatter_x = msdict['scatter_x']
scatter_y = msdict['scatter_y']

color1 = '#7fdada'
color2 = '#FF6961'
color3 = '#e1c58b'

#%% figure4 subplot

def set_custom_yticks(ylim, num_divisions=6, roundn=3, offset=0):
    """ylim을 6분할하여 5개의 tick을 설정하는 함수"""
    ymin, ymax = ylim
    tick_interval = (ymax - ymin) / num_divisions
    yticks = np.round(np.arange(ymin, ymax * 1.001, tick_interval)[1:-1] + offset, roundn)
    return yticks

fig, axs = plt.subplots(1, 2, figsize=(4.36/2 * 3 * 1, 2.45/2 * 2 * 1.4 / 2))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

## A

alpha = 0.2
axs[0].plot(epochs, mean_train_loss, label='Mean Training Loss', c=color1, lw=1)
axs[0].fill_between(epochs, mean_train_loss - \
                       sem_train_loss, mean_train_loss + sem_train_loss, color=color1, alpha=0.8)

axs[0].plot(epochs, mean_val_loss, label='Mean Test Loss', c=color2, lw=1)
axs[0].fill_between(epochs, mean_val_loss - \
                       sem_val_loss, mean_val_loss + sem_val_loss, color=color2, alpha=0.8)

axs[0].set_ylim(0.00, 0.06)
axs[0].set_yticks(set_custom_yticks((0.00, 0.057+0.0001), num_divisions=5, offset=-0.001))
axs[0].set_xlim(-5, 105)
axs[0].set_xticks(np.arange(0, 100 + 1, 20))
axs[0].set_ylabel('Mean Squared Error', fontsize=7, labelpad=0.1)
axs[0].set_xlabel('Epochs', fontsize=7, labelpad=0.5)

#%
## B

axs[1].scatter(scatter_x, scatter_y, alpha=0.5, edgecolors='none', c=color1, s=20)
axs[1].plot(scatter_x, trendline, color=color2, linewidth=2, alpha=0.5)

ylim = (np.round(np.min(scatter_y* 0.98)), np.round(np.max(scatter_y) * 1.02))
xlim = (np.round(np.min(scatter_x* 0.98)), np.round(np.max(scatter_x) * 1.02))

axs[1].set_ylim(ylim)
axs[1].set_yticks(np.arange(650, 1000, 50))
axs[1].set_xlim(xlim)
axs[1].set_xticks(np.arange(730, 950, 50))
axs[1].set_ylabel('Ground Truth RT (ms)', fontsize=7, labelpad=0.1)
axs[1].set_xlabel('Estimated RT (ms)', fontsize=7, labelpad=0.5)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=7, pad=2, width=0.2)  # x tick 폰트 크기 7로 설정
    ax.tick_params(axis='y', labelsize=7, pad=0.2, width=0.2)  # y tick 폰트 크기 7로 설정
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)

if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure4.png'
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()