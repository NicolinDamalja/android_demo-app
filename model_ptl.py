import model
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

if __name__ == "__main__":
    model = model.model_static("data/model_weights.pkl")
    model_dict = model.state_dict()
    snapshot = torch.load("data/model_weights.pkl")
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    model.train(False)
    model.eval()

    example = torch.rand(10, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("data/model.pt")
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter("data/model.ptl")
