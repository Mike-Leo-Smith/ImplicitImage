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


class SIREN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SIREN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 128),
            Sin(),
            nn.Linear(128, 256),
            Sin(),
            nn.Linear(256, 128),
            Sin(),
            nn.Linear(128, 128),
            Sin(),
            nn.Linear(128, out_channels),
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
    print(f"Device: {device}")

    resize = False

    # load dataset
    imread_mode = cv.IMREAD_UNCHANGED
    image = cv.imread("data/color.exr", imread_mode)
    reference = cv.imread("data/reference.exr", imread_mode)
    albedo = cv.imread("data/albedo.exr", imread_mode) * math.pi * 2
    normal = cv.imread("data/normal.exr", imread_mode) * math.pi + math.pi
    shadow = cv.imread("data/visibility.exr", imread_mode) * math.pi * 2
    depth = 1 / (cv.imread("data/depth.exr", imread_mode) + 1) * math.pi * 2
    specular = np.log(cv.imread("data/specular.exr", imread_mode) + 1) * math.pi * 2
    diffuse = np.log(cv.imread("data/diffuse.exr", imread_mode) + 1) * math.pi * 2
    features = [albedo, normal, shadow, depth]
    height, width = image.shape[:2]
    if resize:
        width, height = width // 2, height // 2
        features = [cv.resize(f, (width, height)) for f in features]
        image = cv.resize(image, (width, height))
        reference = cv.resize(reference, (width, height))
    pixel_count = height * width
    image_tm = tonemapping(image)
    reference_tm = tonemapping(reference)
    cv.imshow("Origin", image_tm)
    cv.imshow("Reference", reference_tm)
    cv.waitKey(1)
    coord_scale = torch.from_numpy(np.array([[1 / width, 1 / height]], dtype=np.float32) * math.pi * 2).to(device)
    x_coords = np.transpose(
        np.reshape(np.repeat(np.arange(width, dtype=np.float32), height), [width, height]))
    y_coords = np.reshape(np.repeat(np.arange(height, dtype=np.float32), width), [height, width])
    coords = np.reshape(np.dstack([x_coords, y_coords]), [-1, 2])
    image = torch.log(torch.from_numpy(np.reshape(image, [-1, 3])).to(device) + 1)
    features = [f if len(f.shape) == 3 else f[:, :, np.newaxis] for f in features]
    features = [torch.from_numpy(np.reshape(f, [-1, f.shape[-1]])).to(device) for f in features]
    coords = torch.from_numpy(coords).to(device) * coord_scale

    # create model
    in_channels = 2 + sum(f.shape[-1] for f in features)
    out_channels = 3
    model = SIREN(in_channels, out_channels).to(device)
    loss_fn = nn.MSELoss()
    adam = torch.optim.Adam(model.parameters())
    sgd = torch.optim.SGD(model.parameters(), lr=0.0001)
    summary(model)

    # train
    accum = None
    frame_count = 0
    batch_size = 1024 * 64
    num_epochs = 2048 if resize else 1024
    accum_start = 768 if resize else 384
    for epoch in range(num_epochs):
        indices = torch.randperm(pixel_count).to(device)
        # optimizer = adam if epoch < 1024 else sgd
        optimizer = adam
        for i in range(0, pixel_count, batch_size):
            # jitter = torch.rand_like(coords)
            jitter = torch.zeros_like(coords)
            x = (coords + jitter)[indices[i:i + batch_size]]
            x_features = torch.hstack([x] + [f[indices[i:i + batch_size]] for f in features])
            y = image[indices[i:i + batch_size]]
            pred = model(x_features)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # test
        jitter = torch.rand_like(coords) * coord_scale
        # jitter = torch.zeros_like(coords)
        x_features = torch.hstack([coords + jitter] + features)
        recon = np.reshape(model(x_features).detach().cpu().numpy(), [height, width, 3])
        recon = np.exp(recon) - 1
        if epoch >= accum_start:
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
        abs_diff = cv.absdiff(recon_tm, reference_tm)
        print(f"Epoch #{epoch}: {np.sqrt(np.mean(np.square(np.float32(abs_diff))))}")
        cv.imshow("Error", cv.applyColorMap(abs_diff, cv.COLORMAP_JET))
        cv.waitKey(1)
    cv.imwrite(f"data/result.exr", accum)
    cv.waitKey()
