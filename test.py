# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")