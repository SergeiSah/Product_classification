import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm


def train_mlp_classifier(mlp, dataloaders, criterion, optimizer, param_name, epochs=12, device='cpu'):
    """
    Функция для обучения классификатора.
    """
    history = {'train_loss': [], 'valid_loss': [], 'train_f1': [], 'valid_f1': []}
    best_params = mlp.state_dict()

    for epoch in tqdm(range(1, epochs + 1), desc=f'Train {param_name} classificator. Epochs', leave=False):
        mlp.train()

        epoch_train_loss = []
        y_preds = np.array([])
        y_true = np.array([])

        for X_batch, y_batch in dataloaders['train']:
            X_batch = X_batch.squeeze().to(device)
            y_batch = y_batch.squeeze().to(device)

            optimizer.zero_grad()
            y_pred = mlp(X_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_train_loss.append(loss.item())
            y_preds = np.append(y_preds, y_pred.argmax(dim=1).cpu().numpy())
            y_true = np.append(y_true, y_batch.cpu().numpy())

        history['train_loss'].append(np.mean(epoch_train_loss))
        history['train_f1'].append(f1_score(y_true, y_preds, average='macro'))

        if 'valid' in dataloaders:
            test_loss, f1 = eval_mlp_classifier(mlp, dataloaders['valid'], criterion, device)

            history['valid_loss'].append(test_loss)
            history['valid_f1'].append(f1)

            if epoch > 1 and history['valid_f1'][-1] > history['valid_f1'][-2]:
                best_params = mlp.state_dict()

        elif epoch > 1 and history['train_f1'][-1] > history['train_f1'][-2]:
            best_params = mlp.state_dict()

    return history, best_params


def eval_mlp_classifier(mlp, test_dataloader, criterion, device='cpu'):
    """
    Функция для оценки качества модели на тестовой выборке.
    """
    mlp.eval()

    loss = []
    y_preds = np.array([])
    y_true = np.array([])

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.squeeze().to(device)
            y_batch = y_batch.squeeze().to(device)

            y_pred = mlp(X_batch)

            y_preds = np.append(y_preds, y_pred.argmax(dim=1).cpu().numpy())
            y_true = np.append(y_true, y_batch.cpu().numpy())

            loss.append(criterion(y_pred, y_batch).item())

    return np.mean(loss), f1_score(y_true, y_preds, average='macro')
