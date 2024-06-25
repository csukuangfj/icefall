#!/usr/bin/env python3

import torch
from model import Tcn


def main():
    in_dim = 40
    out_dim = 6
    m = Tcn(feat_dim=in_dim, output_dim=out_dim)
    x = torch.rand(1, 100, in_dim)
    y = m(x)
    print(x.shape, y.shape)


if __name__ == "__main__":
    main()
