import torch
print(torch.cuda.is_available())  # Should return True if CUDA is properly set up
print(torch.cuda.get_device_name(0))  # Should print the name of your GPU
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Please check your setup.")