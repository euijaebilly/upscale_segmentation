import torch


def train(srcnn, unet, data_loader, optimizer_srcnn, optimizer_unet, device='cuda'):
    srcnn.train()
    unet.train()
    total_loss = 0
    for images, annotations in tqdm(data_loader, desc="Training"):
        images = images.to(device)

        optimizer_srcnn.zero_grad()
        restored_images = srcnn(images)
        restored_images = restored_images.to(device)

        optimizer_unet.zero_grad()
        outputs = unet(restored_images)

        loss = ...  # Compute loss using annotations
        loss.backward()
        optimizer_srcnn.step()
        optimizer_unet.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def validate(srcnn, unet, data_loader, device='cuda'):
    srcnn.eval()
    unet.eval()
    total_loss = 0
    with torch.no_grad():
        for images, annotations in tqdm(data_loader, desc="Validating"):
            images = images.to(device)
            restored_images = srcnn(images)
            restored_images = restored_images.to(device)
            outputs = unet(restored_images)

            loss = ...  # Compute loss using annotations

            total_loss += loss.item()

    return total_loss / len(data_loader)
