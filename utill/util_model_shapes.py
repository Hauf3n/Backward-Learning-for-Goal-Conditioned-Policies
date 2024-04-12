import torch

def fw_hook(module, input, output):
    print(f"Shape of output to {module} is {output.shape}.")
        
def model_shapes(model, function_name, input_shape):
    with torch.device("meta"):
        meta_net = model.to("meta")
        inp = torch.randn((input_shape))
        for name, layer in meta_net.named_modules():
            layer.register_forward_hook(fw_hook)
        function_to_call = getattr(meta_net, function_name)
        function_to_call(inp)
