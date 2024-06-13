import torch
import copy
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm


def save_checkpoint(state, filename):
    torch.save(state, filename)


def plot_results(epoch, outputs, labels, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(outputs.cpu().detach().numpy().transpose(1, 2, 0))
    axes[0].set_title("Output Image")
    axes[1].imshow(labels.cpu().detach().numpy().transpose(1, 2, 0))
    axes[1].set_title("Ground Truth")
    plt.savefig(os.path.join(output_dir, f'checkpoint_results_epoch_{epoch}.png'))
    plt.close()


def train_model(srcnn, unet, train_loader, val_loader, criterion, optimizer_srcnn, optimizer_unet, num_epochs=25,
                patience=5, checkpoint_dir='checkpoints'):
    best_model_wts = copy.deepcopy(unet.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        srcnn.train()
        unet.train()

        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.cuda(), labels.cuda()

            # SRCNN
            optimizer_srcnn.zero_grad()
            outputs_srcnn = srcnn(inputs)
            loss_srcnn = criterion(outputs_srcnn, inputs)
            loss_srcnn.backward()
            optimizer_srcnn.step()

            # U-Net
            optimizer_unet.zero_grad()
            outputs_unet = unet(outputs_srcnn)
            loss_unet = criterion(outputs_unet, labels)
            loss_unet.backward()
            optimizer_unet.step()

            running_loss += loss_unet.item() * inputs.size(0)
            progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset))

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validation
        val_loss = 0.0
        srcnn.eval()
        unet.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs_srcnn = srcnn(inputs)
                outputs_unet = unet(outputs_srcnn)
                loss = criterion(outputs_unet, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        # 체크포인트 저장 및 결과 플롯
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'srcnn_state_dict': srcnn.state_dict(),
                'unet_state_dict': unet.state_dict(),
                'optimizer_srcnn_state_dict': optimizer_srcnn.state_dict(),
                'optimizer_unet_state_dict': optimizer_unet.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # 결과 플롯 저장
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs_srcnn = srcnn(inputs)
                outputs_unet = unet(outputs_srcnn)
                plot_results(epoch + 1, outputs_unet[0], labels[0], checkpoint_dir)
                break

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(unet.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    # 최적의 모델 가중치 로드
    unet.load_state_dict(best_model_wts)
    return unet
