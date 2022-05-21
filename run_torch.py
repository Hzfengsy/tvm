import torch
import torchvision.models as models
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import BertConfig, BertModel

DEVICE = "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert DEVICE == "cpu" or DEVICE == "cuda"


def _profile(name, model, inputs):
    model = model.to(DEVICE)
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        inputs = [x.to(DEVICE) for x in inputs]
    else:
        inputs = inputs.to(DEVICE)
    model.eval()

    activities = [ProfilerActivity.CPU] if DEVICE == "cpu" else [ProfilerActivity.CUDA]

    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(1000):
                model(inputs)

    print("workloads", name)
    print(prof.key_averages().table(sort_by=f"{DEVICE}_time_total", row_limit=10))


def C1D():
    n, l, ci, co, kernel, stride, padding = 1, 256, 64, 128, 3, 2, 1
    inputs = torch.randn(n, ci, l, dtype=torch.float32)
    model = nn.Conv1d(ci, co, kernel, stride, padding)
    _profile("C1D", model, inputs)


def C2D():
    n, h, w, ci, co, kernel, stride, padding = 1, 224, 224, 3, 64, 7, 2, 3
    inputs = torch.randn(n, ci, h, w, dtype=torch.float32)
    model = nn.Conv2d(ci, co, kernel, stride, padding)
    _profile("C2D", model, inputs)


def C3D():
    n, d, h, w, ci, co, kernel, stride, padding = 1, 16, 224, 224, 3, 64, 7, 2, 3
    inputs = torch.randn(n, ci, d, h, w, dtype=torch.float32)
    model = nn.Conv3d(ci, co, kernel, stride, padding)
    _profile("C3D", model, inputs)


def CAP():
    b, m, n, k = 1, 128, 128, 128
    inputs = torch.randn(b, m, k, dtype=torch.float32)
    model = nn.Linear(k, n)
    _profile("GMM", model, inputs)


def DEP():
    n, h, w, c, kernel, stride, padding, factor = 1, 112, 112, 32, 3, 1, 1, 1
    inputs = torch.randn(n, c, h, w, dtype=torch.float32)
    model = nn.Conv2d(c, c * factor, kernel, stride, padding, groups=c)
    _profile("DEP", model, inputs)


def DIL():
    n, h, w, ci, co, kernel, stride, padding, dilation = 1, 224, 224, 3, 64, 7, 2, 3, 2
    inputs = torch.randn(n, ci, h, w, dtype=torch.float32)
    model = nn.Conv2d(ci, co, kernel, stride, padding, dilation=dilation)
    _profile("DIL", model, inputs)


def GMM():
    b, m, n, k = 1, 128, 128, 128
    inputs = torch.randn(b, m, k, dtype=torch.float32)
    model = nn.Linear(k, n)
    _profile("GMM", model, inputs)


def GRP():
    n, h, w, ci, co, kernel, stride, padding, groups = 1, 56, 56, 64, 128, 3, 2, 1, 4
    inputs = torch.randn(n, ci, h, w, dtype=torch.float32)
    model = nn.Conv2d(ci, co, kernel, stride, padding, groups=groups)
    _profile("GRP", model, inputs)


def T2D():
    n, h, w, ci, co, kernel, stride, padding = 1, 4, 4, 512, 256, 4, 2, 1
    inputs = torch.randn(n, ci, h, w, dtype=torch.float32)
    model = nn.ConvTranspose2d(ci, co, kernel, stride, padding)
    _profile("T2D", model, inputs)


def C2D_BN_ReLU():
    n, h, w, ci, co, kernel, stride, padding = 1, 224, 224, 3, 64, 7, 2, 3
    inputs = torch.randn(n, ci, h, w, dtype=torch.float32)
    model = nn.Sequential(
        nn.Conv2d(ci, co, kernel, stride, padding),
        nn.BatchNorm2d(co),
        nn.ReLU(),
    )
    _profile("C2D_BN_ReLU", model, inputs)


def TBG():
    b, seq, head, dim = 1, 128, 12, 64
    query = torch.randn(b, seq, head, dim, dtype=torch.float32)
    value = torch.randn(b, seq, head, dim, dtype=torch.float32)

    class TGBModule(nn.Module):
        def forward(self, inputs):
            query, value = inputs
            # shape b, head, seq, dim
            query_T = torch.permute(query, (0, 2, 1, 3))
            # shape b, head, dim, seq
            value_T = torch.permute(value, (0, 2, 3, 1))
            return torch.matmul(query_T, value_T)

    _profile("TBG", TGBModule(), (query, value))


def NRM():
    b, m, n = 1, 256, 256
    inputs = torch.randn(b, m, n, dtype=torch.float32)

    class NRMModule(nn.Module):
        def forward(self, inputs):
            out = torch.norm(inputs, dim=(1, 2), p=2)
            return out

    _profile("NRM", NRMModule(), inputs)


def SFM():
    m, n = 256, 256
    inputs = torch.randn(m, n, dtype=torch.float32)
    model = nn.Softmax(dim=1)
    _profile("SFM", model, inputs)


def resnet():
    inputs = torch.rand((1, 3, 224, 224), dtype=torch.float32)
    model = models.resnet50(pretrained=True)
    _profile("resnet", model, inputs)


def mobilenetV2():
    inputs = torch.rand((1, 3, 224, 224), dtype=torch.float32)
    model = models.mobilenet_v2(pretrained=True)
    _profile("mobilenetV2", model, inputs)


def bert_base():
    configuration = BertConfig(
        num_hidden_layers=12,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        return_dict=False,
    )

    model = BertModel(configuration)
    inputs = torch.randint(10000, (1, 64), dtype=torch.int64)
    _profile("bert_base", model, inputs)



if __name__ == "__main__":
    C1D()
    C2D()
    C3D()
    DEP()
    DIL()
    GMM()
    GRP()
    T2D()
    C2D_BN_ReLU()
    TBG()
    NRM()
    SFM()
    # resnet()
    # mobilenetV2()
    # bert_base()