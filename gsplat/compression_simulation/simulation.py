from dataclasses import dataclass
from hmac import new
from turtle import pos
from typing import Dict, Optional, Tuple, Union, Callable, Literal

from sympy import Union, true
import torch
import torch.nn.functional as F
from torch import Tensor, device

from .ops import fake_quantize_ste, log_transform, inverse_log_transform
from .entropy_model import Entropy_factorized, Entropy_factorized_optimized, Entropy_factorized_optimized_refactor, Entropy_gaussian
from gsplat.compression_simulation import entropy_model

use_clamp = False

class CompressionSimulation:
    """
    """
    def __init__(self, entropy_model_enable: bool = False,
                 entropy_model_type: Literal["factorized_model", "gaussian_model"] = "factorized_model",
                 entropy_steps: Dict[str, int] = None, 
                 device: device = None, 
                 ada_mask_opt: bool = False,
                 ada_mask_step: int = 10_000,
                 ada_mask_strategy: Optional[str] = "learnable",
                 **kwargs) -> None:
        self.entropy_model_enable = entropy_model_enable
        self.entropy_model_type = entropy_model_type
        self.entropy_steps = entropy_steps
        self.device = device

        # TODO: 明确写一个函数，根据splats里有哪些元素，以及配置文件中让哪些元素参与simulation
        self.simulation_option = {
            "means": False,
            "scales": True,
            "quats": True,
            "opacities": True,
            "sh0": True,
            "shN": True
        }
        self.shN_qat = False
        self.shN_ada_mask_opt = ada_mask_opt
        self.shN_ada_mask_step = ada_mask_step
        self.shN_ada_mask_strategy = ada_mask_strategy

        self.q_bitwidth = {
            "means": None,
            "scales": 8,
            "quats": 8,
            "opacities": 8,
            "sh0": 8,
            "shN": None
        }

        self.bds = {
            "means": None,
            "scales": [-10, 2],
            "quats": [-1, 1],
            "opacities": [-15, 15],
            "sh0": [-2, 4],
            "shN": None
        }

        # default settings
        self.entropy_model_option = {
            "means": False,
            "scales": True,
            "quats": True,
            "opacities": False,
            "sh0": True,
            "shN": False
        }

        # turn off if entropy step < 0
        for name, flag in self.entropy_model_option.items():
            if self.entropy_steps[name] < 0:
                 self.entropy_model_option[name] = False

        if self.simulation_option["shN"]:
            if self.shN_qat:
                try:
                    from torchpq.clustering import KMeans
                except:
                    raise ImportError(
                        "Please install torchpq with 'pip install torchpq' to use K-means clustering"
                    )
                n_clusters = 65535
                verbose = True
                self.kmeans = KMeans(n_clusters=n_clusters, distance="manhattan", verbose=verbose)

        # entropy constraint
        if self.entropy_model_enable:
            # entropy model type
            if self.entropy_model_type == "factorized_model":
                self.entropy_models = {
                    "means": None,
                    "scales": Entropy_factorized_optimized_refactor(channel=3, filters=(3, 3)).to(self.device),
                    "quats": Entropy_factorized_optimized_refactor(channel=4).to(self.device),
                    "opacities": None,
                    "sh0": Entropy_factorized_optimized_refactor(channel=3, filters=(3, 3)).to(self.device),
                    "shN": None
                }
            elif self.entropy_model_type == "gaussian_model":
                self.entropy_models = {
                    "means": None,
                    "scales": Entropy_gaussian(channel=3).to(self.device),
                    "quats": Entropy_gaussian(channel=4).to(self.device),
                    "opacities": None,
                    "sh0": Entropy_gaussian(channel=3).to(self.device),
                    "shN": None
                }
            else:
                raise NotImplementedError("Not implemented entropy model type")
            
            # get entropy min step
            selected_key = min((k for k, v in entropy_steps.items() if v > 0), key=lambda k: entropy_steps[k])
            self.entropy_min_step = entropy_steps[selected_key]

            # get corresponding optimizer to optimize params. for prob. estimation
            self.entropy_model_optimizers = {}
            for k, v in self.entropy_models.items():
                # factorized density model: to optimize matrices, biases and factors for learned distribution
                if isinstance(v, Entropy_factorized) or isinstance(v, Entropy_factorized_optimized) or isinstance(v, Entropy_factorized_optimized_refactor):
                    v_opt = torch.optim.Adam(
                        [{"params": p, "lr": 1e-4, "name": n} for n, p in v.named_parameters()]
                    )
                # context-based gaussian distribution model: to optimize hash grid and param. regressor
                elif isinstance(v, Entropy_gaussian):
                    v_opt = torch.optim.Adam(
                        [{'params': v.param_regressor.hash_grid.parameters(), 'lr': 5e-3, "name": 'hash_grid'},
                         {'params': v.param_regressor.mlp_regressor.parameters(), 'lr': 5e-3, "name": 'mlp_regressor'}]
                    )

                else:
                    v_opt = None
                self.entropy_model_optimizers.update({k: v_opt})

            # get scheduler
            self.entropy_model_schedulers = {}
            for k, v in self.entropy_models.items():
                if isinstance(v, Entropy_factorized) or \
                    isinstance(v, Entropy_factorized_optimized) or \
                    isinstance(v, Entropy_factorized_optimized_refactor):
                        # TODO: scheduler for factorized
                        pass
                elif isinstance(v, Entropy_gaussian):
                    v_sch = torch.optim.lr_scheduler.ExponentialLR(
                        self.entropy_model_optimizers[k], gamma=0.01 ** (1.0 / (30000 - entropy_steps[k]))
                    )
                else:
                    v_sch = None
                
                self.entropy_model_schedulers.update({k: v_sch})

        # init. for adaptive mask
        if self.shN_ada_mask_opt:
            if self.shN_ada_mask_strategy == "learnable":
                from .ada_mask import AnnealingMask
                cap_max = kwargs.get("cap_max", 1_000_000)
                self.shN_ada_mask = AnnealingMask(input_shape=[cap_max, 1, 1], 
                                                device=device,
                                                annealing_start_iter=ada_mask_step)
                
                self.shN_ada_mask_optimizer = torch.optim.Adam([
                    {'params': self.shN_ada_mask.parameters(), 'lr': 0.01}
                ])
            elif self.shN_ada_mask_strategy == "gradient":
                pass
            elif self.shN_ada_mask_strategy is None:
                raise ValueError("\'shN_ada_mask_strategy\' should not be None")
            else:
                raise NotImplementedError(f"\'shN_ada_mask_strategy\': {self.shN_ada_mask_strategy} has not been implemented.")


    def _get_simulate_fn(self, param_name: str) -> Callable:
        simulate_fn_map = {
            "means": self.simulate_compression_means,
            "scales": self.simulate_compression_scales,
            "quats": self.simulate_compression_quats,
            "opacities": self.simulate_compression_opacities,
            "sh0": self.simulate_compression_sh0,
            "shN": self.simulate_compression_shN,
        }
        if param_name in simulate_fn_map:
            return simulate_fn_map[param_name]
        else:
            return torch.nn.Identity()

    def _estiblish_bbox(self, pos: Tensor, k: float=1.5):
        # 计算每个维度的四分位数
        self.bbox_lower_bound, self.bbox_upper_bound = torch.quantile(pos, torch.tensor([0.01, 0.99], device=pos.device), dim=0)

    def _get_pts_inside_bbox(self, pos: Tensor) -> Tensor:
        """
        input: pos
        output: mask
        """
        is_inside = torch.all((pos >= self.bbox_lower_bound) & (pos <= self.bbox_upper_bound), dim=1)

        return is_inside
    
    def _random_sample_pts(self, mask: Tensor, ratio: float = 0.05) -> Tensor:
        random_values = torch.rand_like(mask.float())
        random_values[~mask] = 0  # 将invalid位置的概率设为0

        # 选择前ratio%的点
        threshold = torch.quantile(random_values[mask], 1 - ratio)
        return random_values >= threshold


    def simulate_compression(self, splats: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        """
        """
        # Create empty dicts for output, including fake quantized values and (optional) estimated bits
        new_splats = {}
        esti_bits_dict = {}

        # # Randomly sample approximately 5% of the points rather than all points for speedup.
        choose_idx = None
        pos = None
        if self.entropy_model_type == "gaussian_model" and step > self.entropy_min_step:
            inside_mask = self._get_pts_inside_bbox(splats["means"])
            sample_mask = self._random_sample_pts(inside_mask)
            choose_idx = sample_mask
            pos = splats["means"][sample_mask]

        for param_name in splats.keys():
            # Check which params need to be simulate 
            if self.simulation_option[param_name]:
                simulate_fn = self._get_simulate_fn(param_name)
                new_splats[param_name], esti_bits_dict[param_name] = simulate_fn(splats[param_name], step, choose_idx, pos)
            else:
                new_splats[param_name] = splats[param_name] + 0.
                esti_bits_dict[param_name] = None
        
        return new_splats, esti_bits_dict
    
    def simulate_compression_means(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor) -> Tensor:
        # out = torch.clamp(param, -5, 5)
        # out = inverse_log_transform(log_transform(clamped_param))
        
        # return out, None
        return torch.nn.Identity()(param), None
        
    def simulate_compression_quats(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor, pos: torch.Tensor=None) -> Tensor:
        # fake quantize
        if step < 10_000:
            fq_out_dict = fake_quantize_ste(param, self.bds["quats"][0], self.bds["quats"][1], 8)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["quats"][0], self.bds["quats"][1], self.q_bitwidth["quats"])
        
        # entropy constraint
        if step > self.entropy_steps["quats"] and self.entropy_model_enable and self.entropy_model_option["quats"]:
            if choose_idx is not None:
                esti_bits = self.entropy_models["quats"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"], pos=pos)
            else:
                esti_bits = self.entropy_models["quats"](fq_out_dict["output_value"], fq_out_dict["q_step"])

            return fq_out_dict["output_value"], esti_bits
        else:
            return fq_out_dict["output_value"], None

    
    def simulate_compression_scales(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor, pos: torch.Tensor=None) -> Tensor:
        # fake quantize
        if step < 10_000:
            fq_out_dict = fake_quantize_ste(param, self.bds["scales"][0], self.bds["scales"][1], 8)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["scales"][0], self.bds["scales"][1], self.q_bitwidth["scales"])

        # entropy constraint
        if step > self.entropy_steps["scales"] and self.entropy_model_enable and self.entropy_model_option["scales"]:
            # factorized model
            if choose_idx is not None:
                esti_bits = self.entropy_models["scales"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"], pos=pos)
            else:
                esti_bits = self.entropy_models["scales"](fq_out_dict["output_value"], fq_out_dict["q_step"])

            # gaussian model
            # mean = torch.mean(fq_out_dict["output_value"][choose_idx])
            # std = torch.std(fq_out_dict["output_value"][choose_idx])
            # esti_bits = self.entropy_models["scales"](fq_out_dict["output_value"][choose_idx], mean, std, fq_out_dict["q_step"])

            return fq_out_dict["output_value"], esti_bits
        else:
            return fq_out_dict["output_value"], None

    
    def simulate_compression_opacities(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor, pos: torch.Tensor=None) -> Tensor:
        # fake quantize
        fq_out_dict = fake_quantize_ste(param, self.bds["opacities"][0], self.bds["opacities"][1], 8)

        # entropy constraint
        if step > self.entropy_steps["opacities"] and self.entropy_model_enable and self.entropy_model_option["opacities"]:
            fq_out_dict["output_value"] = fq_out_dict["output_value"].unsqueeze(1)
            if choose_idx is not None:
                esti_bits = self.entropy_models["opacities"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"], pos=pos)
            else:
                esti_bits = self.entropy_models["opacities"](fq_out_dict["output_value"], fq_out_dict["q_step"])
            return fq_out_dict["output_value"].squeeze(1), esti_bits
        else:
            return fq_out_dict["output_value"], None


    def simulate_compression_sh0(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor, pos: torch.Tensor=None) -> Tensor:
        # fake quantize
        if step < 10_000:
            fq_out_dict = fake_quantize_ste(param, self.bds["sh0"][0], self.bds["sh0"][1], 8)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["sh0"][0], self.bds["sh0"][1], self.q_bitwidth["sh0"])
        
        # entropy constraint
        if step > self.entropy_steps["sh0"] and self.entropy_model_enable and self.entropy_model_option["sh0"]:
            fq_out_dict["output_value"] = fq_out_dict["output_value"].squeeze(1)
            if choose_idx is not None:
                esti_bits = self.entropy_models["sh0"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"], pos=pos)
            else:
                esti_bits = self.entropy_models["sh0"](fq_out_dict["output_value"], fq_out_dict["q_step"])
            return fq_out_dict["output_value"].unsqueeze(1), esti_bits
        else:
            return fq_out_dict["output_value"], None
    

    def simulate_compression_shN(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor, pos: torch.Tensor=None) -> Tensor:
        if self.shN_ada_mask_opt and self.shN_ada_mask_strategy == "learnable" and step > self.shN_ada_mask_step:
            param = self.shN_ada_mask(param, step)
            # return param, None

        return param, None
    

    def shN_gradient_threshold(self, param: torch.nn.Parameter, step: int) -> None:
        param_value = param.data
        
        # Check which splat's shN are all zero
        zero_mask = (param_value == 0).all(dim=-1).all(dim=-1)

        # Calculate proportion of splats with non-zero shN
        non_zero_mask_ratio = 1 - zero_mask.sum()/zero_mask.size(0)
        # print(f"Vaild shN ratio: {non_zero_mask_ratio*100:.3f}%")

        # Dynamically set threshold based on non-zero ratio,
        gradient_threshold = 2e-3 if non_zero_mask_ratio < 0.10 else 100 # 100, relatively equals to inf

        # Identify gradients below threshold
        low_gradient_mask = torch.norm(param.grad, p=2, dim=(-2,-1)) < gradient_threshold 
        # if step >= 1000:
        #     import pdb; pdb.set_trace()

        # Zero out gradients where both conditions are met
        final_mask = torch.logical_and(zero_mask, low_gradient_mask)
        # final_mask = low_gradient_mask
        param.grad[final_mask] = 0


# STE families
class STE_min_max_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bitdepth = 8):
        maxs = torch.amax(input, dim=0)
        mins = torch.amin(input, dim=0)

        norm_input = (input - mins) / (maxs - mins)
        q_step = 1 / (2**bitdepth - 1)
        q_norm_input = (norm_input / q_step).round() * q_step

        output = q_norm_input * (maxs - mins) + mins

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class STE_quant_for_means(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bitdepth = 16):
        maxs = torch.amax(input, dim=0)
        mins = torch.amin(input, dim=0)

        norm_input = (input - mins) / (maxs - mins)
        q_step = 1 / (2**bitdepth - 1)
        q_norm_input = (norm_input / q_step).round() * q_step

        output = q_norm_input * (maxs - mins) + mins

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class STE_quant_for_quats(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bitdepth = 8):
        # do norm in STE, hope it will not degrade rec. quality
        input = F.normalize(input, dim=-1)

        maxs = 1
        mins = -1

        norm_input = (input - mins) / (maxs - mins)
        q_step = 1 / (2**bitdepth - 1)
        q_norm_input = (norm_input / q_step).round() * q_step

        output = q_norm_input * (maxs - mins) + mins

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class STE_quant_for_scales_clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bitdepth = 8):
        maxs = 0 # 2.5
        mins = -12 # -10

        input = input.clamp_(mins, maxs)

        norm_input = (input - mins) / (maxs - mins)
        q_step = 1 / (2**bitdepth - 1)
        q_norm_input = (norm_input / q_step).round() * q_step

        output = q_norm_input * (maxs - mins) + mins

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class STE_quant_for_scales_min_max_q(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bitdepth = 8):
        maxs = torch.amax(input.detach(), dim=0)
        mins = torch.amin(input.detach(), dim=0)

        norm_input = (input - mins) / (maxs - mins)
        q_step = 1 / (2**bitdepth - 1)
        q_norm_input = (norm_input / q_step).round() * q_step

        output = q_norm_input * (maxs - mins) + mins

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class STE_quant_for_opacities_clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bitdepth = 8):
        maxs = 15
        mins = -15

        input = input.clamp_(mins, maxs)

        norm_input = (input - mins) / (maxs - mins)
        q_step = 1 / (2**bitdepth - 1)
        q_norm_input = (norm_input / q_step).round() * q_step

        output = q_norm_input * (maxs - mins) + mins

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class STE_quant_for_sh0_clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bitdepth = 8):
        maxs = 4
        mins = -2

        input = input.clamp_(mins, maxs)

        norm_input = (input - mins) / (maxs - mins)
        q_step = 1 / (2**bitdepth - 1)
        q_norm_input = (norm_input / q_step).round() * q_step

        output = q_norm_input * (maxs - mins) + mins

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q, input_mean=None):
        Q_round = torch.round(input / Q)
        Q_q = Q_round * Q
        return Q_q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
# to simulate what happens in gsplat 's PngCompression()
def _min_max_quantization_16bit(param: Tensor) -> Tensor:
    maxs = torch.amax(param, dim=0)
    mins = torch.amin(param, dim=0)

    param_norm = (param - mins) / (maxs - mins)
    q_step = 1 / (2**16 - 1)
    q_param_norm = (((param_norm / q_step).round() * q_step) - param_norm).detach() + param_norm

    q_param = q_param_norm * (maxs - mins) + mins

    return q_param

# to simulate what happens in gsplat 's PngCompression()
def _min_max_quantization(param: Tensor) -> Tensor: # seems not working...
    maxs = torch.amax(param, dim=0)
    mins = torch.amin(param, dim=0)

    param_norm = (param - mins) / (maxs - mins)
    q_step = 1 / (2**8 - 1)
    q_param_norm = (((param_norm / q_step).round() * q_step) - param_norm).detach() + param_norm

    q_param = q_param_norm * (maxs - mins) + mins

    return q_param

def _ste_quantization_for_quats(param: Tensor) -> Tensor:
    return STE_quant_for_quats.apply(param, 8)

def _ste_quantization_given_q_step(param: Tensor) -> Tensor:
    return STE_multistep.apply(param, 0.001)

def _ste_only(param: torch.nn.Parameter) -> torch.nn.Parameter:
    return param
    # return STE.apply(param)
    # return (param.detach() -  param.detach()) + param # not working...

def _add_noise_to_simulate_quantization(param: Tensor) -> Tensor:
    return param + torch.empty_like(param).uniform_(-0.5, 0.5) * 0.001