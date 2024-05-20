import matplotlib.pyplot as plt


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