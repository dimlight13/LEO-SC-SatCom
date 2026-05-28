import os


def configure_cuda_visible_devices(default_device="1"):
    """Restrict TensorFlow to one physical CUDA device before GPU init."""
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("SC_SATCOM_CUDA_DEVICE", default_device)
