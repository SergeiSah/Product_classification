import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


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


def train_ruclip_one_epoch(clip, dataloader, loss_img, loss_txt, optimizer, device='cpu'):

    embeddings = torch.zeros((len(dataloader.dataset), 1024), dtype=torch.float32)
    losses = []

    clip.train()
    for batch in tqdm(dataloader, desc='Batch', leave=False, total=len(dataloader)):
        optimizer.zero_grad()

        idxs, pixel_values, input_ids = batch

        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)

        image_features = clip.encode_image(pixel_values)
        text_features = clip.encode_text(input_ids)

        # normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        concat = torch.cat((image_features, text_features), dim=1).detach().cpu()
        for i, idx in enumerate(idxs):
            embeddings[idx] = concat[i]

        # cosine similarity as logits
        logit_scale = clip.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(len(pixel_values), dtype=torch.long, device=device)

        img_loss = loss_img(logits_per_image, labels)
        txt_loss = loss_txt(logits_per_text, labels)

        loss = (img_loss + txt_loss) / 2
        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    clip.eval()
    return embeddings, np.mean(losses)
