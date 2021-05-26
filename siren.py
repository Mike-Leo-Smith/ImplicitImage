import cv2 as cv
import numpy as np
import torch
from torch import nn
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
            nn.Linear(2, 256),
            Sin(),
            nn.Linear(256, 128),
            Sin(),
            nn.Linear(128, 64),
            Sin(),
            nn.Linear(64, 16),
            Sin(),
            nn.Linear(16, 3))

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            PositionalEncoding(16),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.ReLU())

    def forward(self, x):
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

    # load dataset
    image = np.clip(cv.imread("data/reference.exr", cv.IMREAD_UNCHANGED), 0, 1)
    height, width = image.shape[:2]
    image = cv.resize(image, (width // 4, height // 4))
    image_tm = tonemapping(image)
    cv.imshow("Origin", image_tm)
    cv.waitKey(1)
    height, width = image.shape[:2]
    pixel_count = height * width
    x_coords = np.transpose(np.reshape(np.repeat(np.arange(width, dtype=np.float32), height), [width, height]))
    y_coords = np.reshape(np.repeat(np.arange(height, dtype=np.float32), width), [height, width])
    coords = np.reshape(np.dstack([x_coords, y_coords]), [-1, 2])
    image = np.reshape(image, [-1, 3])
    dataset = torch.from_numpy(np.float32(np.hstack([coords, image]))).to(device)

    # create model
    model = SIREN().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train
    taa = None
    batch_size = 256
    indices = np.arange(pixel_count)
    for epoch in range(4096):
        np.random.shuffle(indices)
        for i in range(0, pixel_count, batch_size):
            batch = dataset[indices[i:i + batch_size]]
            x, y = batch[:, :2], batch[:, 2:]
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # test
        recon = np.reshape(model(torch.from_numpy(coords)).detach().numpy(), [height, width, 3])
        taa = recon if epoch == 0 else 0.8 * taa + 0.2 * recon
        taa_tm = tonemapping(taa)
        cv.imshow("Reconstruct", taa_tm)
        abs_diff = cv.absdiff(taa_tm, image_tm)
        print(f"Epoch #{epoch}: {np.sqrt(np.mean(np.square(np.float32(abs_diff))))}")
        cv.imshow("Difference", cv.applyColorMap(abs_diff, cv.COLORMAP_JET))
        cv.waitKey(1)
        cv.imwrite(f"data/test-{epoch}.exr", taa)
