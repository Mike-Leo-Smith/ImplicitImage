import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu, sigmoid


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class PositionEncoding(nn.Module):
    def __init__(self, n, e, scale) -> None:
        super().__init__()
        self.num_inputs = n
        self.num_encoding_functions = e
        self.features = 2 * e + self.num_inputs
        self.B = torch.randn((self.num_encoding_functions,
                              self.num_inputs)).to(device) * scale

    def forward(self, x):
        enc = [x.T]
        # print(self.B.shape, x.shape)
        enc.append((torch.sin(2.0 * 3.1415926535 * self.B @ x.T)))
        enc.append((torch.cos(2.0 * 3.1415926535 * self.B @ x.T)))
        x = torch.cat(enc).to(device).T
        return x


class SIREN(nn.Module):
    def __init__(self):
        super(SIREN, self).__init__()
        self.normal_enc = PositionEncoding(3, 64, 10.0)
        self.albedo_enc = PositionEncoding(3, 64, 10.0)
        self.coord_enc = PositionEncoding(2, 128, 10.0)
        # self.num_inputs = 2 + 3 + 3
        # self.num_encoding_functions = 20
        # self.input_features = 2 * \
        #     self.num_encoding_functions + self.num_inputs
        # self.B = torch.randn((self.num_encoding_functions,  self.num_inputs)).to(device) * 10.0
        self.input_features = self.coord_enc.features + \
            self.normal_enc.features + self.albedo_enc.features
        # self.layers = nn.Sequential(
        #     nn.Linear(self.input_features, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 3),
        #     nn.Sigmoid())
        self.layer0 = nn.Linear(self.input_features, 64)
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64+64, 64)
        self.layer5 = nn.Linear(64+64, 64)
        self.layer6 = nn.Linear(64+64, 3)

    def forward(self, x):
        x = x.to(device)
        coord = x[:, :2]
        albedo = x[:, 2:2+3]
        normal = x[:, 2+3:]
        e1 = self.normal_enc(normal)
        e2 = self.coord_enc(coord)
        e3 = self.albedo_enc(albedo)
        x = torch.cat([e1.T, e2.T, e3.T]).to(device).T

        out0 = relu(self.layer0(x))
        out1 =  relu(self.layer1(out0))
        out2 =  relu(self.layer2(out1))
        out3 =  relu(self.layer3(out2))
        out =  relu(self.layer4(torch.cat([out2.T, out3.T]).T))
        out =  relu(self.layer5(torch.cat([out1.T, out.T]).T))
        # print(out.shape, out0.shape)
        out =  self.layer6(torch.cat([out0.T, out.T]).T)
        return sigmoid(out)


def rgb2srgb(image):
    return np.uint8(np.round(np.clip(np.where(
        image <= 0.00304,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055
    ) * 255, 0, 255)))


def tonemapping(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return rgb2srgb(x * (a * x + b) / (x * (c * x + d) + e))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # load dataset
    color = cv.imread("data/color.exr", cv.IMREAD_UNCHANGED)
    normal = cv.imread("data/normal.exr", cv.IMREAD_UNCHANGED)
    albedo = cv.imread("data/albedo.exr", cv.IMREAD_UNCHANGED)
    # print(color, normal)
    height, width = color.shape[:2]
    width //= 2
    height //= 2
    color = np.clip(cv.resize(color, (width, height)), 0.0, 1.0)
    normal = cv.resize(normal, (width, height))
    albedo = cv.resize(albedo, (width, height))
    # print(color.shape, normal.shape, albedo.shape)

    # imcolorage = cv.resize(color, (width // 4, height // 4))

    color_tm = tonemapping(color)
    cv.imshow("Origin", color_tm)
    cv.waitKey(1)
    height, width = color.shape[:2]
    # color = cv.resize(color, (width // 2, height //  ))

    pixel_count = height * width
    x_coords = np.transpose(np.reshape(
        np.repeat(np.arange(width, dtype=np.float32), height), [width, height]))
    y_coords = np.reshape(
        np.repeat(np.arange(height, dtype=np.float32), width), [height, width])
    coords = np.reshape(np.dstack([x_coords, y_coords]), [-1, 2])

    # reference = np.zeros_like(color)
    # for i in range(3):
    # reference = cv.bilateralFilter(src=color, d=5,sigmaColor=50, sigmaSpace=75)
    # reference[:, 0] = cv.bilateralFilter(src=color[:,0], d=5,sigmaColor=50, sigmaSpace=75)
    # reference[:, 1] = cv.bilateralFilter(src=color[:,1], d=5,sigmaColor=50, sigmaSpace=75)
    # reference[:, 2] = cv.bilateralFilter(src=color[:,2], d=5,sigmaColor=50, sigmaSpace=75)

    color = np.reshape(color, [-1, 3])
    normal = np.reshape(normal, [-1, 3])
    albedo = np.reshape(albedo, [-1, 3])
    # reference = np.reshape(reference, [-1, 3])

    dataset = torch.from_numpy(np.float32(
        np.hstack([coords, albedo, normal, color]))).to(device)

    # create model
    model = SIREN().to(device)
    loss_fn = nn.MSELoss().to(device)
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def lr_f(epoch):
        global learning_rate
        if epoch % 50 == 50 - 1 and learning_rate > 0.0001:
            learning_rate *= 0.9
        print(learning_rate)
        return learning_rate

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_f)
    # train
    batch_size = 1024 * 40
    # indices = np.arange(pixel_count)
    # indices = torch.arange(pixel_count).to(device)

    taa = [np.zeros((height, width, 3)) for _ in range(4)]
    # taa_alpha = 0.3
    for epoch in range(4096):
        print(f"Epoch #{epoch}")

        # torch.random.shuffle(indices)
        indices = torch.randperm(pixel_count).to(device)
        for i in range(0, pixel_count, batch_size):
            batch = dataset[indices[i:i + batch_size]].to(device)
            x, y = batch[:, :2+3+3], batch[:, 2+3+3:]
            # print(batch.device, x.device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            del x, y
        # scheduler.step()
        print(loss.item())
        # print(dataset.device, indices.device)
        # test
        recon = np.reshape(model(torch.from_numpy(
            np.hstack([coords, albedo, normal]))).detach().cpu().numpy(), [height, width, 3])
        # recon = taa_alpha * recon + (1.0 - taa_alpha) * taa
        taa.pop()
        taa.insert(0, recon)
        if epoch >= 500:
            recon = sum(taa) / len(taa)
        recon_tm = tonemapping(recon)
        cv.imshow("Reconstruct", recon_tm)
        cv.imshow("Difference", cv.applyColorMap(
            cv.absdiff(recon_tm, color_tm), cv.COLORMAP_JET))
        cv.waitKey(1)
        #
    cv.imwrite(f"data/test-{epoch}.exr", recon)
