import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from sklearn.manifold import TSNE
from pandas import DataFrame
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

# pd.options.display.notebook_repr_html=False  # 表格显示
# plt.rcParams['figure.dpi'] = 400  # 图形分辨率
# sns.set_theme(style='darkgrid')  # 图形主题

# # target acc with different self-tarining Loss weight
# x=[0.0,0.2,0.4,0.6,0.8,1.0]
# y=[76.59,92.74,92.85,92.29,91.62,91.51]
# plt.ylim((70,95))
# plt.plot(x,y,lw=2.5,marker='o')
# plt.xlabel('Self-Training Loss weight')
# plt.ylabel('Top-1 Target Accuracy')
# #plt.show()
# plt.savefig('/media/sobremesa/E/DACO_Best_NanQI/DACO/experiment/New/office31/Acc_selftrain_weight.pdf')

# # target acc with different self-tarining Loss weight
# x=[0.0,0.2,0.4,0.6,0.8,1.0]
# y=[89.50,92.52,91.85,92.74,92.96,92.85]
# plt.ylim((80,96))
# plt.plot(x,y,lw=2.5,marker='o')
# plt.xlabel('Self-Supervised Loss weight')
# plt.ylabel('Top-1 Target Accuracy')
# #plt.show()
# plt.savefig('/media/sobremesa/E/DACO_Best_NanQI/DACO/experiment/New/office31/Acc_selfsupervised_weight.pdf')


# # target acc with different filter ini
# x=[0.1,0.2,0.3,0.4,0.5]
# y=[91.40,92.74,91.96,90.17,91.62]
# plt.ylim((85,95))
# plt.plot(x,y,lw=2.5,marker='o')
# plt.xlabel('Initial filter threshold')
# plt.ylabel('Top-1 Target Accuracy')
# #plt.show()
# plt.savefig('/media/sobremesa/E/DACO_Best_NanQI/DACO/experiment/New/office31/filter_threshold.pdf')


# # Target Acc with different methods and w/o sf w/o st
# plt.figure(figsize=(8,7))
# # pr 0.09 a2w
# x=[0.09,0.16,0.22,0.29]
# # DACO
# y = [91.85, 89.95, 89.06, 86.04]# 1.0 0.6 
# # w/o self-training
# y_1 = [78.90, 78.15, 78.57, 79.12]# 0.0 0.6
# # w/o self-supervised
# #y_2 = [88.95, 89.17, 89.50, 82.14] # 1.0 0.0 original 因为根据s supervised得消融图可以看出来看，ss loss=0.6时效果最差，因此不能说加了ss loss就是副作用
# y_2 = [86.95, 87.17, 87.50, 80.14]

# # other methods
# #PiCO CycleDA CGDM RCRDP	
# y_PiCO = [69.53, 71.65, 68.36, 65.72]
# y_DPLL = [60.37, 60.88, 58.36, 67.63]


# plt.ylim((30,100))
# plt.plot(x,y,lw=2.3,marker='o',ls='-',label='DACO')
# plt.plot(x,y_1,lw=2.3,marker='*',markersize=10,ls='-.',label='DACO w/o self-training')
# plt.plot(x,y_2,lw=2.3,marker='p',ls='-.',label='DACO w/o self-supervised')
# plt.plot(x,y_PiCO,lw=2.3,marker='o',ls='-',label='PiCO')
# plt.plot(x,y_DPLL,lw=2.3,marker='o',ls='-',label='RCRDP')


# plt.xlabel('Partial Label rate')
# plt.ylabel('Top-1 Target Accuracy(%)')
# plt.legend(prop={'size': 9})
# #plt.show()
# plt.savefig('/media/sobremesa/E/DACO_Best_NanQI/DACO/experiment/Ablation_Study/comparison_diff_pr.jpg')


# pseudo-labels curve
# The best parameters st: 0.2 ss: 0.6



# # T-SNE

# features = '/media/sobremesa/E/DACO_Best_NanQI/DACO/experiment/New/feature_ablation/featuresT(A2W)_0.2(nossp).npy'
# features = np.load(features)
# # Art
# target_file = '/media/sobremesa/E/DACO_Best_NanQI/DACO/data/office31/webcam.txt'
# with open(target_file, 'r') as f:
#     file_dir, true_labels = [], []
#     for i in f.read().splitlines():
#         file_dir.append(i.split(' ')[0])
#         true_labels.append(int(i.split(' ')[1]))
# Y = np.array(true_labels)
# X_embedded = TSNE(n_components=2).fit_transform(features)
# data = np.column_stack((X_embedded, Y))
# df = DataFrame(data, columns=['DIM_1', 'DIM_2', 'Label'])
# df = df.astype({'Label':'int'})
# df.dtypes
# sns.set_context('notebook', font_scale=2.)
# #sns.set_style("darkgrid")
# fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(10,8))
# # fig 1
# sns.scatterplot(
#     x='DIM_1',
#     y='DIM_2',
#     hue='Label',
#     data=df,
#     palette='plasma',
#     ax=ax,
#     edgecolor='none'
# )
# # # fig 2
# # sns.scatterplot(
# #     x='DIM_1',
# #     y='DIM_2',
# #     data=df,
# #     hue='Label',
# #     palette='viridis',
# #     ax=ax1
# # )
# norm = plt.Normalize(0, 30)
# sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
# sm.set_array([])
# ax.get_legend().remove()
# #ax1.get_legend().remove()
# ax.figure.colorbar(sm)
# plt.xlabel('T-SNE Dimension 1').set_fontsize('20')
# plt.ylabel('T-SNE Dimension 2').set_fontsize('20')
# plt.show()
# plt.savefig("/media/sobremesa/E/DACO_Best_NanQI/DACO/experiment/New/feature_ablation/fea_TSNE_pr02(noselfsuper).jpg")

# # Class-wise Accuracy 柱状图形式
# pd.options.display.notebook_repr_html=False  # 表格显示
# plt.rcParams['figure.dpi'] = 400  # 图形分辨率
# sns.set_theme(style='darkgrid')  # 图形主题
# plt.figure(figsize=(25, 10))
# x=np.arange(31)
# #y=[100.0, 100.0, 96.42857142857143, 100.0, 100.0, 100.0, 100.0, 94.44444444444444, 90.47619047619048, 100.0, 100.0, 100.0, 93.33333333333333, 89.47368421052632, 80.0, 67.44186046511628, 96.66666666666667, 96.29629629629629, 100.0, 78.125, 93.75, 95.0, 70.0, 77.77777777777779, 67.5, 90.9090909090909, 100.0, 100.0, 95.83333333333334, 95.65217391304348, 90.47619047619048]
# # pr 0.16
# y=[100.0, 100.0, 96.42857142857143, 100.0, 100.0, 100.0, 100.0, 94.44444444444444, 90.47619047619048, 100.0, 100.0, 100.0, 93.33333333333333, 89.47368421052632, 80.0, 67.44186046511628, 96.66666666666667, 96.29629629629629, 100.0, 78.125, 93.75, 95.0, 70.0, 77.77777777777779, 67.5, 90.9090909090909, 100.0, 100.0, 95.83333333333334, 95.65217391304348, 90.47619047619048]
# y_noba=[100.0, 100.0, 85.71428571428571, 100.0, 100.0, 100.0, 100.0, 100.0, 90.47619047619048, 100.0, 100.0, 100.0, 93.33333333333333, 68.42105263157895, 3.3333333333333335, 65.11627906976744, 100.0, 100.0, 100.0, 100.0, 93.75, 85.0, 50.0, 48.148148148148145, 52.5, 45.45454545454545, 100.0, 70.0, 95.83333333333334, 100.0, 85.71428571428571]

# # pr 0.2
# # y=[100.0, 100.0, 85.71428571428571, 100.0, 100.0, 100.0, 100.0, 100.0, 90.47619047619048, 100.0, 100.0, 100.0, 96.66666666666667, 78.94736842105263, 90.0, 62.7906976744186, 100.0, 96.29629629629629, 100.0, 81.25, 93.75, 95.0, 80.0, 92.5925925925926, 67.5, 90.9090909090909, 100.0, 100.0, 79.16666666666666, 100.0, 100.0]
# # y_noba=[100.0, 100.0, 85.71428571428571, 100.0, 100.0, 100.0, 100.0, 100.0, 90.47619047619048, 100.0, 100.0, 100.0, 93.33333333333333, 84.21052631578947, 3.3333333333333335, 62.7906976744186, 100.0, 100.0, 100.0, 100.0, 93.75, 85.0, 50.0, 44.44444444444444, 60.0, 45.45454545454545, 100.0, 76.66666666666667, 95.83333333333334, 82.6086956521739, 80.95238095238095]

# plt.ylim((0,105))
# plt.bar(x,y,align='center',label='Ours',width=0.4)
# plt.bar(x+0.4,y_noba,align='center',label='w/o balance',width=0.4)

# x_name = ['back_pack','bike','bike_helmet','bookcase','bottle','calculator','desk_chair','desk_lamp','desktop_computer','file_cabinet','headphones','keyboard','laptop_computer','letter_tray','mobile_phone','monitor','mouse','mug','paper_notebook','pen','phone','printer','projector','punchers','ring_binder','ruler','scissors','speaker','stapler','tape_dispenser','trash_can'] # 类名字竖排
# # 设置x轴的说明
# plt.xlabel('Classes',size=18,weight='bold')
# # 设置y轴的说明
# plt.ylabel('Class-wise Accuracy',size=18,weight='bold')
# plt.xticks(ticks=x, labels=x_name,rotation=90,size=17)
# plt.yticks(size=17)
# plt.legend()
# plt.show()
# plt.savefig('/media/sobremesa/E/DACO_Best_NanQI/DACO/experiment/New/office31/Class_Wise_Acc_{}.jpg'.format(0.16), bbox_inches='tight')
