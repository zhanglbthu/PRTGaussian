import commentjson as json
import tinycudann as tcnn
import torch

with open("config/config_hash.json") as f:
	config = json.load(f)

n_input_dims = 2
n_output_dims = 243

encoding = tcnn.Encoding(n_input_dims, config["encoding"])
network = tcnn.Network(encoding.n_output_dims, n_output_dims, config["network"])
decoder = torch.nn.Sequential(encoding, network)