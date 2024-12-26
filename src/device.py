import torch

type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
dtype = torch.float16
