#%%

# #b4c7e7 - 파랑
# #f4b183 - 주황
# #a9d18e - 초록

import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

#%%

# pickle 파일 불러오기
spath = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure' + '\\'
with open(spath + 'vnstroop_figure2.pickle', 'rb') as file:
    msdict = pickle.load(file)

# 변수명으로 다시 할당
x = msdict['x']
width = msdict['width']
sham_means_figure3a = msdict['sham_means_figure3a']
sham_sems_figure3a = msdict['sham_sems_figure3a']
vns_means_figure3a = msdict['vns_means_figure3a']
vns_sems_figure3a = msdict['vns_sems_figure3a']
sham_means_figure3c = msdict['sham_means_figure3c']
sham_sems_figure3c = msdict['sham_sems_figure3c']
vns_means_figure3c = msdict['vns_means_figure3c']
vns_sems_figure3c = msdict['vns_sems_figure3c']
dict_name = msdict['dict_name']
sessions = msdict['sessions']
msdict_save = msdict['msdict_save']

print("변수들이 성공적으로 불러와졌습니다.")



#%% figure2A
fig, ax = plt.subplots(figsize=(4.36/2, 2.45/2))  # Convert cm to inches

# Adjust the position of the plot area to take up most of the figure
ax.set_position([0.1, 0.1, 0.85, 0.85])  # [left, bottom, width, height]

# Bar width and x positions
width = 0.35
x = np.array(range(len(sham_means_figure3c)))

# Plotting bars
bars1 = ax.bar(x - width/2, sham_means_figure3a, width, yerr=sham_sems_figure3a, label='Sham', \
               capsize=2, color='#B7FFF4', error_kw=dict(lw=0.6, capthick=0.6))
    
bars2 = ax.bar(x + width/2, vns_means_figure3a, width, yerr=vns_sems_figure3a, label='VNS', \
               capsize=2, color='#E2DCFF', error_kw=dict(lw=0.6, capthick=0.6))


ax.set_xticks(x)
ax.set_xticklabels(x + 1, fontsize=7)
ax.tick_params(width=0.2)  # Adjust the value for the desired thickness of the ticks

ax.set_ylim([-0.3, 1])  # Replace xmin and xmax with your desired range
ax.set_yticks(np.arange(-0.2, 1.1, 0.2))  # Replace start, stop, step with your desired values
ax.tick_params(axis='y', labelsize=7)  # Replace 7 with the desired font size

for spine in ax.spines.values():
    spine.set_linewidth(0.2) 
    
if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure2A.png'
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    
#%% figure2C

fig, ax = plt.subplots(figsize=(4.36/2, 2.45/2))  # Convert cm to inches

# Adjust the position of the plot area to take up most of the figure
ax.set_position([0.1, 0.1, 0.85, 0.85])  # [left, bottom, width, height]

# Bar width and x positions
width = 0.35
x = np.array(range(len(sham_means_figure3c)))

# Plotting bars
bars1 = ax.bar(x - width/2, sham_means_figure3c, width, yerr=sham_sems_figure3c, label='Sham', \
               capsize=2, color='#B7FFF4', error_kw=dict(lw=0.6, capthick=0.6))
    
bars2 = ax.bar(x + width/2, vns_means_figure3c, width, yerr=vns_sems_figure3c, label='VNS', \
               capsize=2, color='#E2DCFF', error_kw=dict(lw=0.6, capthick=0.6))

ax.set_xlabel('Session #', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(x + 1, fontsize=7)
ax.tick_params(width=0.2)  # Adjust the value for the desired thickness of the ticks

ax.set_ylim([0, 1])  # Replace xmin and xmax with your desired range
ax.set_yticks(np.arange(0, 1.1, 0.2))  # Replace start, stop, step with your desired values
ax.tick_params(axis='y', labelsize=7)  # Replace 7 with the desired font size

for spine in ax.spines.values():
    spine.set_linewidth(0.2) 

if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure2C.png'
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    

#%% figure 2B


fig, axs = plt.subplots(2, 3, figsize=(4.36/2 * 3 * 0.7, 2.45/2 * 2 * 1.4))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
# plt.subplots_adjust(hspace=0.6)  # 기본값은 0.2입니다. 0.6으로 늘려서 간격을 더 넓힙니다.


for ix, n in enumerate(dict_name):
    vns_data, sham_data = [], []
    for se in sessions:
        vns = msdict_save[n][str(se)]['VNS']
        sham = msdict_save[n][str(se)]['sham']
        
        vns_data.extend(vns)
        sham_data.extend(sham)
        
        # t_stat, p_value = stats.ttest_ind(sham, vns)
        # print()
        # print(n, se)
        # print("t-statistic:", t_stat)
        # print("p-value:", p_value)

    # Combine data for violin plot
    all_data = []
    group_labels = []
    session_labels = []

    for i, se in enumerate(sessions):
        all_data.extend(msdict_save[n][str(se)]['sham'])
        group_labels.extend(['Sham'] * len(msdict_save[n][str(se)]['sham']))
        session_labels.extend([f'{i+1}'] * len(msdict_save[n][str(se)]['sham']))
        
        all_data.extend(msdict_save[n][str(se)]['VNS'])
        group_labels.extend(['VNS'] * len(msdict_save[n][str(se)]['VNS']))
        session_labels.extend([f'{i+1}'] * len(msdict_save[n][str(se)]['VNS']))

    # 현재 subplot에 대한 ax 선택
    ax = axs[ix // 3, ix % 3]
    ax.tick_params(width=0.2)  # Adjust the value for the desired thickness of the ticks
    
    # Define colors
    sham_color = '#B7FFF4'
    vns_color = '#E2DCFF'
    sham_scatter_color = '#7CC7C7'
    vns_scatter_color = '#A794C3'
    sham_violin_line_color = '#B7FFF4'
    vns_violin_line_color = '#E2DCFF'

    # Plotting violin plot with specified colors
    sns.violinplot(x=session_labels, y=all_data, hue=group_labels, split=True, inner=None, ax=ax,
                   palette={'Sham': sham_color, 'VNS': vns_color}, alpha=1, linewidth=0, legend=False)

        
    if ix in [0]: ax.set_ylabel('Millisecond', fontsize=7)
    if ix in [3]: ax.set_ylabel('False ratio', fontsize=7)
        
    if ix in [3,4,5]:
        ax.set_xlabel('Session #', fontsize=7)
    # ax.set_title(n, fontsize=9)
    
    if ix in [0,1,2]:
        ax.set_ylim([300, 1200])  # Replace xmin and xmax with your desired range
        ax.set_yticks(np.arange(300, 1401, 200))
        
    if ix in [3,4,5]:
        ax.set_ylim([-0.2, 0.8])  # Replace xmin and xmax with your desired range
        ax.set_yticks(np.arange(0, 0.9, 0.2)) 
        
    for ax in axs.flat:
        ax.tick_params(axis='x', labelsize=7)  # x tick 폰트 크기 10으로 설정
        ax.tick_params(axis='y', labelsize=7)  # y tick 폰트 크기 10으로 설정

for ax in axs.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)
        
if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure2B.png'
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)

    
#%%

# 저장된 pickle 파일 불러오기
with open('vnstroop_figure3.pickle', 'rb') as file:
    msdict = pickle.load(file)

# 변수명으로 다시 할당
target_freqs_05 = msdict['target_freqs_05']
psd_pre_median_sham = msdict['psd_pre_median_sham']
psd_pre_sem_sham = msdict['psd_pre_sem_sham']
psd_post_median_sham = msdict['psd_post_median_sham']
psd_post_sem_sham = msdict['psd_post_sem_sham']
psd_pre_median_VNS = msdict['psd_pre_median_VNS']
psd_pre_sem_VNS = msdict['psd_pre_sem_VNS']
psd_post_median_VNS = msdict['psd_post_median_VNS']
psd_post_sem_VNS = msdict['psd_post_sem_VNS']
median_psd_3b_sham = msdict['median_psd_3b_sham']
sem_psd_3b_sham = msdict['sem_psd_3b_sham']
median_psd_3b_vns = msdict['median_psd_3b_vns']
sem_psd_3b_vns = msdict['sem_psd_3b_vns']
g1_means = msdict['g1_means']
g1_sems = msdict['g1_sems']
g0_means = msdict['g0_means']
g0_sems = msdict['g0_sems']
bands = msdict['bands']
figure3D_plot_data = msdict['figure3D_plot_data']
sham_means = msdict['sham_means']
sham_sems = msdict['sham_sems']
vns_means = msdict['vns_means']
vns_sems = msdict['vns_sems']
delta__beta = msdict['delta__beta']
delta_tscore = msdict['delta_tscore']
trendline = msdict['trendline']

print("변수들이 성공적으로 불러와졌습니다.")


def set_custom_yticks(ylim, num_divisions=6, roundn=3, offset=0):
    """ylim을 6분할하여 5개의 tick을 설정하는 함수"""
    ymin, ymax = ylim
    tick_interval = (ymax - ymin) / num_divisions
    yticks = np.round(np.arange(ymin, ymax * 1.001, tick_interval)[1:-1] + offset, roundn)
    return yticks

fig, axs = plt.subplots(2, 3, figsize=(4.36/2 * 3 * 1.2, 2.45/2 * 2 * 1.4))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# 1,1 자리에 해당하는 subplot에 plot하기 (Python 인덱스 기준으로 0,0 위치)

## A

alpha = 0.2
axs[0, 0].plot(target_freqs_05, psd_pre_median_sham, label='First Interval', c='#73E0D1', lw=1)
axs[0, 0].fill_between(target_freqs_05, psd_pre_median_sham - \
                       psd_pre_sem_sham, psd_pre_median_sham + psd_pre_sem_sham, alpha=alpha, color='#73E0D1')

axs[0, 0].plot(target_freqs_05, psd_post_median_sham, label='Second Interval', c='#49C4B1', lw=1)
axs[0, 0].fill_between(target_freqs_05, psd_post_median_sham - \
                       psd_post_sem_sham, psd_post_median_sham + psd_post_sem_sham, alpha=alpha, color='#49C4B1')

axs[0, 0].plot(target_freqs_05, psd_pre_median_VNS, label='First Interval_vns', c='#C7B6FF', lw=1)
axs[0, 0].fill_between(target_freqs_05, psd_pre_median_VNS - \
                       psd_pre_sem_VNS, psd_pre_median_VNS + psd_pre_sem_VNS, alpha=alpha, color='#C7B6FF')

axs[0, 0].plot(target_freqs_05, psd_post_median_VNS, label='Second Interval_vns', c='#A182FF', lw=1)
axs[0, 0].fill_between(target_freqs_05, psd_post_median_VNS - \
                       psd_post_sem_VNS, psd_post_median_VNS + psd_post_sem_VNS, alpha=alpha, color='#A182FF')

axs[0, 0].set_ylim(0.01, 0.045)
axs[0, 0].set_yticks(set_custom_yticks((0.01, 0.045), num_divisions=5))
axs[0, 0].set_xlim(0, 40)
axs[0, 0].set_xticks(np.arange(0, 40 + 1, 10))
axs[0, 0].set_ylabel('Relative Power', fontsize=7, labelpad=0.1)
axs[0, 0].set_xlabel('Frequency (Hz)', fontsize=7, labelpad=0.5)

## B

axs[0, 1].plot(target_freqs_05, median_psd_3b_sham, label='Sham', c='#73E0D1', lw=1)
axs[0, 1].fill_between(target_freqs_05, median_psd_3b_sham - sem_psd_3b_sham, \
                       median_psd_3b_sham + sem_psd_3b_sham, alpha=alpha, color='#73E0D1')

axs[0, 1].plot(target_freqs_05, median_psd_3b_vns, label='VNS', c='#C7B6FF', lw=1)
axs[0, 1].fill_between(target_freqs_05, median_psd_3b_vns - sem_psd_3b_vns, \
                       median_psd_3b_vns + sem_psd_3b_vns, alpha=alpha, color='#C7B6FF')
    
axs[0, 1].set_ylim(-3.5, 2.5)
axs[0, 1].set_yticks(set_custom_yticks((-4, 2.5), num_divisions=5, offset=0.1))
axs[0, 1].set_xlim(0, 40)
axs[0, 1].set_xticks(np.arange(0, 40 + 1, 10))
axs[0, 1].set_ylabel('Relative Power Difference', fontsize=7, labelpad=0.1)
axs[0, 1].set_xlabel('Frequency (Hz)', fontsize=7, labelpad=0.5)

## C

x = np.arange(4)
width = 0.4
axs[0, 2].bar(x - width/2, g1_means, width, yerr=g1_sems, capsize=2, label='sham', color='#B7FFF4', error_kw=dict(lw=0.6, capthick=0.6))
axs[0, 2].bar(x + width/2, g0_means, width, yerr=g0_sems, capsize=2, label='VNS', color='#E2DCFF', error_kw=dict(lw=0.6, capthick=0.6))

axs[0, 2].set_ylim(-3, 1.7)
axs[0, 2].set_yticks(set_custom_yticks((-3, 1.5), num_divisions=5, offset=0.3))
axs[0, 2].set_xticks(x)
axs[0, 2].set_xticklabels(bands)
axs[0, 2].set_ylabel('Mean of Power Difference', fontsize=7, labelpad=0.1)

## D
# Sham (Original): 밝은 청록색 (#B7FFF4)
# VNS (Original): 밝은 연보라색 (#E2DCFF)
# Sham (New): 진한 청록색 (#73E0D1)
# VNS (New): 진한 연보라색 (#C7B6FF)
# Sham (Extended): 더욱 진한 청록색 (#49C4B1)
# VNS (Extended): 더욱 진한 보라색 (#A182FF)
                           
colors1 = ['#B7FFF4', '#73E0D1', '#49C4B1']
colors2 = ['#E2DCFF', '#C7B6FF', '#A182FF']
alpha = 0.2
xaxis = np.arange(4, 40, dtype=int)
fi_ix = slice(4, 40)
for t in range(3):
    mean_sham = figure3D_plot_data[t][0]
    sem_sham = figure3D_plot_data[t][2]
    axs[1, 0].plot(xaxis, mean_sham[fi_ix], c=colors1[t], label='sham_' + str(t), lw=1)
    axs[1, 0].fill_between(xaxis, mean_sham[fi_ix] - sem_sham[fi_ix], mean_sham[fi_ix] + sem_sham[fi_ix], color=colors1[t], alpha=alpha)
    
    mean_vns = figure3D_plot_data[t][1]
    sem_vns = figure3D_plot_data[t][3]
    axs[1, 0].plot(xaxis, mean_vns[fi_ix], c=colors2[t], label='VNS_' + str(t), lw=1)
    axs[1, 0].fill_between(xaxis, mean_vns[fi_ix] - sem_vns[fi_ix], mean_vns[fi_ix] + sem_vns[fi_ix], color=colors2[t], alpha=alpha)

axs[1, 0].set_ylim(0.0, 0.027)
axs[1, 0].set_yticks(set_custom_yticks((0.0, 0.027), num_divisions=5, offset=0))
axs[1, 0].set_xlim(0, 44)
axs[1, 0].set_xticks(np.arange(0, 40 + 1, 10))
axs[1, 0].set_ylabel('Relative Power', fontsize=7, labelpad=0.1)
axs[1, 0].set_xlabel('Frequency (Hz)', fontsize=7, labelpad=0.5)
## D

# 바 그래프 그리기
labels = ['Session 1', 'Session 2', 'Session 3']
x = np.arange(len(labels))  # x축 위치
width = 0.35  # 막대 너비

axs[1, 1].bar(x - width/2, sham_means, width, yerr=sham_sems, label='Sham', color='#B7FFF4', capsize=2, error_kw=dict(lw=0.6, capthick=0.6))
axs[1, 1].bar(x + width/2, vns_means, width, yerr=vns_sems, label='VNS', color='#E2DCFF', capsize=2, error_kw=dict(lw=0.6, capthick=0.6))

axs[1, 1].set_ylim(0.008, 0.018)
axs[1, 1].set_yticks(set_custom_yticks((0.008, 0.018), num_divisions=5, offset=0))
# axs[1, 0].set_xlim(4, len(mean_vns))
axs[1, 1].set_ylabel('Mean of Relative Power', fontsize=7, labelpad=0.1)
axs[1, 1].set_xticks(x)
axs[1, 1].set_xlabel('Sessions #', fontsize=7, labelpad=0.5)
axs[1, 1].set_xticklabels(range(1,4))

## F
axs[1, 2].plot(delta__beta, trendline, color='lightcoral', linewidth=1, linestyle='-', label='Trendline', alpha=0.7)
axs[1, 2].scatter(delta__beta, delta_tscore, s=7, c='#A182FF', alpha=0.7, edgecolors='none')

axs[1, 2].set_ylim(-0.2, 0.9)
axs[1, 2].set_yticks(set_custom_yticks((-0.2, 0.9), num_divisions=5, offset=-0.02))
axs[1, 2].set_ylabel('Δ Stroop Score', fontsize=7, labelpad=0.1)

axs[1, 2].set_xlim(-0.35, 0.55)
axs[1, 2].set_xticks(np.arange(-0.3, 0.5 * 1.00001, 0.2))
axs[1, 2].set_xlabel('Δ Beta Power', fontsize=7, labelpad=0.5)


###    
for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=7, pad=2, width=0.2)  # x tick 폰트 크기 7로 설정
    ax.tick_params(axis='y', labelsize=7, pad=0.2, width=0.2)  # y tick 폰트 크기 7로 설정
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)

if True:
    fsave2 = 'Figure3.png'
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()

#%% Figure 4 - data load

with open('vnstroop_figure4.pickle', 'rb') as file:
    msdict = pickle.load(file)

print(msdict.keys())
epochs = msdict['epochs']
mean_train_loss = msdict['mean_train_loss']
sem_train_loss = msdict['sem_train_loss']
mean_val_loss = msdict['mean_val_loss']
sem_val_loss = msdict['sem_val_loss']
scatter_x = msdict['scatter_x']
scatter_y = msdict['scatter_y']
trendline = msdict['trendline']



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
    # fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = 'Figure4.png'
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()