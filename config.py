import os

i = 0
seed = 22
epochs = 115
batch_sizes = [16, 32, 64, 128]
dropout_rates = [0.2, 0.3, 0.4, 0.5, 0.6]
normalizations = ["batch", "layer", "group", "instance", "local_response"]
poolings = ["max_pooling", "average_pooling"]

def creation_file_name(directory, file_name):
    return os.path.join(directory, file_name)

for batch_size in batch_sizes:
    for dropout_rate in dropout_rates:
        for normalization in normalizations:
            for pooling in poolings:

                i += 1
                file_name = creation_file_name("config", f"config_{i}.yaml")
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, "w", encoding="utf-8") as f:
                    content = f"""seed: {seed}
epochs: {epochs}
batch_size: {batch_size}
dropout: {dropout_rate}
normalization: {normalization}
pooling: {pooling}
"""
                    f.write(content.strip())
                print(f"Finished with config_file_{i}")
