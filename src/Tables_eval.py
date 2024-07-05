
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from collections import OrderedDict
sb.set()

names_cl = ['KNN','SVM','Decision Tree', 'Random Forest', 'MLP', 'NB']
names_cl = ['SVM','MLP','NB']
names_scores = ['precision', 'recall','f1-score']
list_corr_size = [4,10,20,50]
cl_scores = OrderedDict()
for cl in names_cl:
    cl_score = []
    for corr in list_corr_size:
        pp = pd.read_csv(r'score_f1_'+str(corr)+'_autocorr_'+cl+'_rgbt.csv', header=[0]).values[:,0]
        for p, line in enumerate(pp):
            line = line.split(',')
            score = []
            for sco in range(len(names_scores)):
                score += [float(line[sco].split(' ')[-1])]
            line_ = pd.DataFrame([score],columns=names_scores)
            corr_score = pd.DataFrame([score], columns=names_scores) if p == 0 else pd.concat([corr_score, line_], ignore_index=True)
        list_scores, new_names = [], []
        for name in (names_scores):
            mean_score = np.round(corr_score[name].mean()*100,2)
            std_score = np.round(corr_score[name].std()*100,2)
            list_scores += [mean_score]
            list_scores += [std_score]
            new_names += [name+'_mean']
            new_names += [name+'_std']
        cl_ = pd.DataFrame([list_scores],index=[corr],columns=new_names)

        cl_score = cl_.copy() if corr == list_corr_size[0] else pd.concat([cl_score, cl_])
    cl_scores[cl] = cl_score


labels = ['4', '10', '20', '50']
sb.set(font_scale=4)
gen_clr = ['#9CCC78','#FFAC63','#C05746','#439A86','#007991','#9CAFB7']
recal_clr = ['#3c5c23','#803c00','#723127','#17352f','#00414d','#36434a']
prec_clr = ['#e4f1da','#ffd6b3','#ebcbc6','#cae8e2','#b3f4ff','#c4ced4']

names = ["SVM","MLP","NB"]
x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
widths = [-0.25,-0.15,-0.05,0.05,0.15,0.25]
fig, ax = plt.subplots(figsize=(30,15))
rgbt_names = pd.DataFrame()
rgbt_names_dev = pd.DataFrame()
for i, n in enumerate(names):
    print(n)
    vals = cl_scores[n]
    #plt.scatter([0,1,2,3,4],[92.8,94.6,94.2,93.7,88.1],s=500, color=clr[0])
    curr_x = x + widths[i]
    rects1 = ax.bar(curr_x, 1,width/5, vals['f1-score_mean'].values, yerr = vals['f1-score_std'].values, label=n, color=gen_clr[i])
    #rects1 = ax.bar(curr_x, 1, width / 5, vals['accuracy'].values, yerr=vals['accuracy'].values, label=n,
     #               color=gen_clr[i])

    #rects1 = ax.bar(curr_x,1, width/5, vals['recall_mean'].values, yerr = vals['recall_std'].values, label=n, color=recal_clr[i])


    #rects1 = ax.bar(curr_x,1, width/5, vals['precision_mean'].values, yerr = vals['precision_std'].values, label=n, color=prec_clr[i])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores (%)')
ax.set_title('RGBT dataset results for ')
ax.set_xticks(x)
# ax.set_ylim(66,96)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.1, 0.95))
plt.show()
fig.savefig('Results_f1_RGBT.eps', bbox_inches='tight',  format='eps')




