import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle


colors = cycle([
    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
    (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
    (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
    (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
    (0.8, 0.7254901960784313, 0.4549019607843137),
    (0.39215686274509803, 0.7098039215686275, 0.803921568627451)
])


def plot_history(history, char_name=None):
    """
    Функция для отрисовки графиков обучения MLP классификаторов.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = axes.flatten()

    if char_name:
        fig.suptitle(f'{char_name}')

    axes[0].plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='train')
    if 'valid_loss' in history:
        axes[0].plot(range(1, len(history['train_f1']) + 1), history['valid_loss'], label='valid')
    axes[0].set_title('Cross Entropy Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    if 'train_f1' in history:
        axes[1].plot(range(1, len(history['train_f1']) + 1), history['train_f1'], label='train')
        if 'valid_f1' in history:
            axes[1].plot(range(1, len(history['train_f1']) + 1), history['valid_f1'], label='valid')
        axes[1].set_title('F1-macro score')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('F1 score')
        axes[1].legend()

    plt.show()

    return fig


def plot_clusters(labels, reduced_embed, add_info=''):
    df = pd.DataFrame({'x': reduced_embed[:, 0], 'y': reduced_embed[:, 1], 'label': labels})

    fig = plt.figure()
    for label in df.label.unique():
        plt.scatter(df[df.label == label].x, df[df.label == label].y, s=1)

    plt.title(f'{labels.nunique()} clusters' + add_info)
    plt.show()

    return fig


__all__ = [
    'plot_history',
    'plot_clusters'
]
