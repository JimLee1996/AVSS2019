import torch
from model import VioNet


class Backend:

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = VioNet(pretrained=True).to(self.device)
        self.net.eval()  # important

    # input tensor with (batch_size)x3x16x112x112
    def predict(self, x):
        x = x.to(self.device)
        # need N x C x T x H x W
        if len(x.shape) < 5:
            x = x.unsqueeze(0)
        y = self.net(x)

        _, results = y.topk(1, 1, True)

        labels = ['normal' if x else 'violent' for x in results]

        return labels


# test code
if __name__ == '__main__':
    be = Backend()
    x = torch.randn(3, 16, 112, 112)
    print(be.predict(x))

    x = torch.randn(2, 3, 16, 112, 112)
    print(be.predict(x))
