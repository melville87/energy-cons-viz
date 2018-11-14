import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import category_encoders as ce


def get_mapping(encoder):

    ''' Prints encoder mapping of categorical features '''

    for item in encoder.category_mapping:
        print(item["col"], ":", item["mapping"])


def get_cls_metrics(cls_name, scores):

    ''' Calculates the median scores for the given classifier,
        Plots the distribution of each score on resamples (cv iterations)
        using a boxplot. '''

    dash = "-"
    print(6*dash)
    print(4*dash, cls_name, 4*dash)
    print(6*dash)

    for name in scores.keys():
        print(name, "{:.3f}".format(np.median(scores[name])))

    names = [ key for key in scores.keys() if key not in ["fit_time", "score_time"] ]

    fig, axes = plt.subplots(1, 3, figsize= (12, 4))
    plt.subplots_adjust(wspace= 0.5)

    for i in range(3):
        plt.subplot(1, 3, i+1)
        axes[i] = sns.boxplot(y= scores[names[i]], width= 0.3)
        axes[i] = plt.title(names[i])
        axes[i] = plt.ylim(0, 1.0)
