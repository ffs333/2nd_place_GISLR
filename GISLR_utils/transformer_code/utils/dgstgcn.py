import torch
import torch.nn as nn
import torch.nn.functional as F


class DGGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.num_groups = A.shape[0]
        self.A = nn.Parameter(A.clone()[None, :, None]) # 1 (B), K, 1 (C), V, V
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, T, V = x.shape

        time_pool_x = x.mean(-2, keepdim = True) # B x C x 1 x V
        x1 = self.conv1(time_pool_x).reshape(B, self.num_groups, -1, V) # B x K x (C/K) x V
        x2 = self.conv2(time_pool_x).reshape(B, self.num_groups, -1, V)
        ctr_graph = (x1.unsqueeze(-1) - x2.unsqueeze(-2)).tanh() # B x K x (C/K) x V x V
        ada_graph = torch.einsum('nkcv,nkcw->nkvw', x1, x2).unsqueeze(2).softmax(-2) # B x K x 1 x V x V
        A = self.A + ctr_graph * self.alpha + ada_graph * self.beta # B x K x (C/K) x V x V

        res = self.residual(x)
        x = self.relu(self.stem(x)).reshape(B, self.num_groups, -1, T, V) # B x K x (C/K) x T x V
        x = torch.einsum('nkctv,nkcvw->nkctw', x, A).reshape(B, -1, T, V)
        return self.relu(self.head(x) + res)

class DGMSTCN(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride = 1):

        super().__init__()
        num_branches = 4
        mid_channels = out_channels // num_branches

        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1),
                          nn.BatchNorm2d(mid_channels),
                          nn.ReLU(),
                          UnitTCN(mid_channels, mid_channels, 3, stride = stride, dilation = 1, norm = False)),
            # nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1),
            #               nn.BatchNorm2d(mid_channels),
            #               nn.ReLU(),
            #               UnitTCN(mid_channels, mid_channels, 3, stride = stride, dilation = 2, norm = False)),
            nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1),
                          nn.BatchNorm2d(mid_channels),
                          nn.ReLU(),
                          UnitTCN(mid_channels, mid_channels, 3, stride = stride, dilation = 3, norm = False)),
            # nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1),
            #               nn.BatchNorm2d(mid_channels),
            #               nn.ReLU(),
            #               UnitTCN(mid_channels, mid_channels, 3, stride = stride, dilation = 4, norm = False)),
            nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1),
                          nn.BatchNorm2d(mid_channels),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size = (3, 1), stride = (stride, 1), padding = (1, 0))),
            nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, stride = (stride, 1))),
        ])

        
        self.gamma = nn.Parameter(torch.zeros(A.shape[1]))
        self.head = nn.Sequential(
            nn.BatchNorm2d(mid_channels * num_branches), 
            nn.ReLU(), 
            nn.Conv2d(mid_channels * num_branches, out_channels, kernel_size = 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = torch.cat([x, x.mean(-1, keepdim = True)], -1)
        out = torch.cat([branch(x) for branch in self.branches], dim = 1)
        out = out[...,:-1] + torch.einsum('nct,v->nctv', out[...,-1], self.gamma)
        out = self.head(out)
        return out

class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, dilation = 1, norm = True):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = (kernel_size, 1),
            padding = (padding, 0),
            stride = (stride, 1),
            dilation = (dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))

class DGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride = 1, residual = True):
        super().__init__()
        self.gcn = DGGCN(in_channels, out_channels, A)
        self.tcn = DGMSTCN(out_channels, out_channels, A, stride = stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = None
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size = 1, stride = stride)

    def forward(self, x):
        out = self.gcn(x)
        out = self.tcn(out)
        if self.residual is not None:
            out += self.residual(x)
        return self.relu(out)

class DGSTGCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        A = torch.rand(cfg.num_groups, cfg.num_nodes, cfg.num_nodes) * 0.02 + 0.04
        self.pre_norm = nn.BatchNorm1d(cfg.in_channels * cfg.num_nodes)

        self.gcns = nn.Sequential(
            DGBlock(cfg.in_channels,       cfg.base_channels,     A.clone(), 1, residual = False),
            # DGBlock(cfg.base_channels,     cfg.base_channels,     A.clone(), 1),
            DGBlock(cfg.base_channels,     cfg.base_channels,     A.clone(), 1),
            # DGBlock(cfg.base_channels,     cfg.base_channels,     A.clone(), 1),
            DGBlock(cfg.base_channels,     cfg.base_channels * 2, A.clone(), 2),
            DGBlock(cfg.base_channels * 2, cfg.base_channels * 2, A.clone(), 1),
            # DGBlock(cfg.base_channels * 2, cfg.base_channels * 2, A.clone(), 1),
            DGBlock(cfg.base_channels * 2, cfg.base_channels * 4, A.clone(), 2),
            # DGBlock(cfg.base_channels * 4, cfg.base_channels * 4, A.clone(), 1),
            DGBlock(cfg.base_channels * 4, cfg.base_channels * 4, A.clone(), 1),
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.drop_rate),
            nn.Linear(cfg.base_channels * 4, cfg.num_labels)
        )

    def forward(self, x):
        B, V, C, T = x.shape
        x = self.pre_norm(x.reshape(B, V * C, T))
        x = x.reshape(B, V, C, T).permute(0, 2, 3, 1) # B x C x T x V
        
        x = self.gcns(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # from torchsummary import summary
    # summary(model)
    class CFG:
        in_channels = 9
        base_channels = 64
        num_nodes = 37
        num_groups = 8
        num_labels = 250
        drop_rate = 0.1
    cfg = CFG()
    model = DGSTGCN(cfg)

    x = torch.rand(128, 37, 9, 96)
    y = model(x)
    y.shape

    torch.onnx.export(
        model,
        x,
        "tmp.onnx",
        export_params = True,
        keep_initializers_as_inputs = True,
        verbose = False,
        input_names = ['inputs'],
        output_names = ['outputs'],
        dynamic_axes = {"inputs": [0]},
        opset_version = 12
    )

    import onnx
    import onnxruntime as rt
    onnx_model = onnx.load("tmp.onnx")

    sess_options = rt.SessionOptions()
    sess_options.enable_profiling = True
    # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # sess_options.optimized_model_filepath = opt_onnx_model_path
    # sess_options.intra_op_num_threads = 1
    sess = rt.InferenceSession("tmp.onnx", sess_options)
    onnx_result = sess.run(None, {"inputs": x.numpy()})[0]
    prof_file = sess.end_profiling()

    import json
    with open(prof_file, "r") as f:
        p = json.load(f)

    import pandas as pd
    d = pd.DataFrame(p)
    d["op_name"] = [row.args.get("op_name", "None") for _, row in d.iterrows()]

    import polars as pl
    a = pl.from_pandas(d)
    print(a.groupby("op_name").agg(pl.col("dur").sum()).sort("dur").to_pandas().to_string())