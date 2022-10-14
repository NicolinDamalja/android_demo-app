import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.models import MobileNet_V3_Small_Weights

model = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("app/src/main/assets/model.pt")