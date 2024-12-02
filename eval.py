
"""
I want to actually evaluate a model from HF on here.
Against what? Idk. Maybe against the original data?

https://github.com/sasilver75/admire-pipeline/commit/24775fb9fd28b9e08f1e19055fb04ed960790739#diff-aec5f361b4ee3ec65c93339d6f05e37648dc77c733e1fcbe1b316f71300378ff
"""

# TODO: Maybe it also makes sense to upload the original model to our repo?
# Or maybe we can just load it from the Qwen VL repo on HF...
model_name = "UCSC-Admire/Admire-Finetune-2024-12-01_22-21-53"

# TODO: Evaluate against some held-out synthetic data, or... evaluate against the original data?
# The former is proably easier for now...