import cv2 as cv
import numpy as np
import torch
from torch import nn
from torchinfo import summary
import math


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class PositionalEncoding(nn.Module):
    def __init__(self, num_bands):
        super(PositionalEncoding, self).__init__()
        self.num_bands = num_bands

    def forward(self, x):
        bands = [torch.cos(2 ** (i - self.num_bands) * math.pi * x) for i in range(self.num_bands)]
        return torch.hstack(bands)


class SIREN(nn.Module):
    def __init__(self):
        super(SIREN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 128),
            Sin(),
            nn.Linear(128, 256),
            Sin(),
            nn.Linear(256, 128),
            Sin(),
            nn.Linear(128, 128),
            Sin(),
            nn.Linear(128, 3),
            nn.ReLU())

    def forward(self, coords, albedo, normal):
        x = torch.hstack([coords, albedo, normal])
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            PositionalEncoding(16),
            nn.Linear(16 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.ReLU())

    def forward(self, coords, albedo, normal):
        x = torch.hstack([coords, albedo, normal])
        return self.layers(x)


def rgb2srgb(image):
    return np.uint8(np.round(np.clip(np.where(
        image <= 0.00304,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055
    ) * 255, 0, 255)))


def tonemapping(x):
    x = np.maximum(x, 0.0)
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return rgb2srgb(x * (a * x + b) / (x * (c * x + d) + e))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # load dataset
    image = np.clip(cv.imread("data/reference.exr", cv.IMREAD_UNCHANGED), 0, 1)
    albedo = cv.imread("data/albedo.exr", cv.IMREAD_UNCHANGED)
    normal = cv.imread("data/normal.exr", cv.IMREAD_UNCHANGED)
    height, width = image.shape[:2]
    width, height = width // 2, height // 2
    pixel_count = height * width
    image = cv.resize(image, (width, height))
    albedo = cv.resize(albedo, (width, height)) * 10
    normal = cv.resize(normal, (width, height)) * 5 + 5
    image_tm = tonemapping(image)
    cv.imshow("Origin", image_tm)
    cv.waitKey(1)
    x_coords = np.transpose(np.reshape(np.repeat(np.arange(width, dtype=np.float32), height), [width, height]))
    y_coords = np.reshape(np.repeat(np.arange(height, dtype=np.float32), width), [height, width])
    coords = np.reshape(np.dstack([x_coords, y_coords]), [-1, 2])
    image = torch.from_numpy(np.reshape(image, [-1, 3])).to(device)
    albedo = torch.from_numpy(np.reshape(albedo, [-1, 3])).to(device)
    normal = torch.from_numpy(np.reshape(normal, [-1, 3])).to(device)
    coords = torch.from_numpy(coords).to(device)

    # create model
    model = SIREN().to(device)
    loss_fn = nn.MSELoss()
    adam = torch.optim.Adam(model.parameters())
    sgd = torch.optim.SGD(model.parameters(), lr=0.0001)
    summary(model)

    # train
    accum = None
    frame_count = 0
    batch_size = 1024 * 64
    for epoch in range(4096 * 4096):
        indices = torch.randperm(pixel_count).to(device)
        # optimizer = adam if epoch < 1024 else sgd
        optimizer = adam
        for i in range(0, pixel_count, batch_size):
            # jitter = torch.rand_like(coords)
            jitter = torch.zeros_like(coords)
            x, x_albedo, x_normal, y = (coords + jitter)[indices[i:i + batch_size]], \
                                       albedo[indices[i:i + batch_size]], \
                                       normal[indices[i:i + batch_size]], \
                                       image[indices[i:i + batch_size]]
            pred = model(x, x_albedo, x_normal)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # test
        jitter = torch.rand_like(coords)
        # jitter = torch.zeros_like(coords)
        recon = np.reshape(model(coords + jitter, albedo, normal).detach().cpu().numpy(), [height, width, 3])
        if epoch >= 1024:
            # accum = recon if accum is None else 0.8 * accum + 0.2 * recon
            frame_count += 1
            t = 1 / frame_count
            accum = recon if accum is None else (1 - t) * accum + t * recon
            recon = accum
        if (epoch + 1) % 64 == 0:
            batch_size = max(batch_size // 2, 4096)
            print(f"Batch Size: {batch_size}")
        recon_tm = tonemapping(recon)
        cv.imshow("Reconstruct", recon_tm)
        abs_diff = cv.absdiff(recon_tm, image_tm)
        print(f"Epoch #{epoch}: {np.sqrt(np.mean(np.square(np.float32(abs_diff))))}")
        cv.imshow("Difference", cv.applyColorMap(abs_diff, cv.COLORMAP_JET))
        cv.waitKey(1)
        # cv.imwrite(f"data/test-{epoch}.exr", recon)
    cv.waitKey()
