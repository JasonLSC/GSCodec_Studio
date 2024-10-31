from dataclasses import dataclass
from hmac import new
from typing import Dict, Optional, Tuple, Union, Callable

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
                 entropy_steps: Dict[str, int] = None, 
                 device: device = None, 
                 ada_mask_opt: bool = False,
                 ada_mask_step: int = 10_000,
                 **kwargs) -> None:
        self.entropy_model_enable = entropy_model_enable
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

        self.entropy_model_option = {
            "means": False,
            "scales": True,
            "quats": True,
            "opacities": False,
            "sh0": True,
            "shN": False
        }

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

        if self.entropy_model_enable:
            self.entropy_models = {
                "means": None,
                # "scales": Entropy_factorized(channel=3).to(self.device),
                # "quats": Entropy_factorized(channel=4).to(self.device),
                "scales": Entropy_factorized_optimized_refactor(channel=3, filters=(3, 3)).to(self.device),
                "quats": Entropy_factorized_optimized_refactor(channel=4).to(self.device),
                "opacities": Entropy_factorized(channel=1).to(self.device),
                "sh0": Entropy_factorized_optimized_refactor(channel=3, filters=(3, 3)).to(self.device),
                "shN": None
            }

            self.entropy_model_optimizers = {}
            for k, v in self.entropy_models.items():
                if isinstance(v, Entropy_factorized) or isinstance(v, Entropy_factorized_optimized) or isinstance(v, Entropy_factorized_optimized_refactor):
                    v_opt = torch.optim.Adam(
                        [{"params": p, "lr": 1e-4, "name": n} for n, p in v.named_parameters()]
                    )
                    # v_opt = torch.optim.SGD(
                    #     [{"params": p, "lr": 1e-4, "name": n} for n, p in v.named_parameters()]
                    # )
                else:
                    v_opt = None
                self.entropy_model_optimizers.update({k: v_opt})

        if self.shN_ada_mask_opt:
            from .ada_mask import AnnealingMask
            cap_max = kwargs.get("cap_max", 1_000_000)
            self.shN_ada_mask = AnnealingMask(input_shape=[cap_max, 1, 1], 
                                              device=device,
                                              annealing_start_iter=ada_mask_step)
            
            self.shN_ada_mask_optimizer = torch.optim.Adam([
                {'params': self.shN_ada_mask.parameters(), 'lr': 0.01}
            ])

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

    def simulate_compression(self, splats: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        """
        """
        # Create empty dicts for output, including fake quantized values and (optional) estimated bits
        new_splats = {}
        esti_bits_dict = {}

        # # Randomly sample approximately 5% of the points rather than all points for speedup.
        # choose_idx = torch.rand_like(splats["means"][:, 0], device=self.device) <= 1
        choose_idx = None

        for param_name in splats.keys():
            # Check which params need to be simulate 
            if self.simulation_option[param_name]:
                simulate_fn = self._get_simulate_fn(param_name)
                new_splats[param_name], esti_bits_dict[param_name] = simulate_fn(splats[param_name], step, choose_idx)
            else:
                new_splats[param_name] = splats[param_name] + 0.
                esti_bits_dict[param_name] = None
        
        return new_splats, esti_bits_dict
    
    def simulate_compression_means(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor) -> Tensor:
        # out = torch.clamp(param, -5, 5)
        # out = inverse_log_transform(log_transform(clamped_param))
        
        # return out, None
        return torch.nn.Identity()(param), None
        
    def simulate_compression_quats(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor) -> Tensor:
        # fake quantize
        if step < 10_000:
            fq_out_dict = fake_quantize_ste(param, self.bds["quats"][0], self.bds["quats"][1], 8)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["quats"][0], self.bds["quats"][1], self.q_bitwidth["quats"])
        
        # entropy constraint
        if step > self.entropy_steps["quats"] and self.entropy_model_enable and self.entropy_model_option["quats"]:
                if choose_idx is not None:
                    esti_bits = self.entropy_models["quats"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"])
                else:
                    esti_bits = self.entropy_models["quats"](fq_out_dict["output_value"], fq_out_dict["q_step"])

                return fq_out_dict["output_value"], esti_bits
        else:
            return fq_out_dict["output_value"], None

    
    def simulate_compression_scales(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor) -> Tensor:
        # fake quantize
        if step < 10_000:
            fq_out_dict = fake_quantize_ste(param, self.bds["scales"][0], self.bds["scales"][1], 8)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["scales"][0], self.bds["scales"][1], self.q_bitwidth["scales"])

        # entropy constraint
        if step > self.entropy_steps["scales"] and self.entropy_model_enable and self.entropy_model_option["scales"]:
            # factorized model
            if choose_idx is not None:
                esti_bits = self.entropy_models["scales"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"])
            else:
                esti_bits = self.entropy_models["scales"](fq_out_dict["output_value"], fq_out_dict["q_step"])

            # gaussian model
            # mean = torch.mean(fq_out_dict["output_value"][choose_idx])
            # std = torch.std(fq_out_dict["output_value"][choose_idx])
            # esti_bits = self.entropy_models["scales"](fq_out_dict["output_value"][choose_idx], mean, std, fq_out_dict["q_step"])

            return fq_out_dict["output_value"], esti_bits
        else:
            return fq_out_dict["output_value"], None

    
    def simulate_compression_opacities(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor) -> Tensor:
        # fake quantize
        fq_out_dict = fake_quantize_ste(param, self.bds["opacities"][0], self.bds["opacities"][1], 8)

        # entropy constraint
        if step > self.entropy_steps["opacities"] and self.entropy_model_enable and self.entropy_model_option["opacities"]:
            fq_out_dict["output_value"] = fq_out_dict["output_value"].unsqueeze(1)
            if choose_idx is not None:
                esti_bits = self.entropy_models["opacities"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"])
            else:
                esti_bits = self.entropy_models["opacities"](fq_out_dict["output_value"], fq_out_dict["q_step"])
            return fq_out_dict["output_value"].squeeze(1), esti_bits
        else:
            return fq_out_dict["output_value"], None


    def simulate_compression_sh0(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor) -> Tensor:
        # fake quantize
        if step < 10_000:
            fq_out_dict = fake_quantize_ste(param, self.bds["sh0"][0], self.bds["sh0"][1], 8)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["sh0"][0], self.bds["sh0"][1], self.q_bitwidth["sh0"])
        
        # entropy constraint
        if step > self.entropy_steps["sh0"] and self.entropy_model_enable and self.entropy_model_option["sh0"]:
            fq_out_dict["output_value"] = fq_out_dict["output_value"].squeeze(1)
            if choose_idx is not None:
                esti_bits = self.entropy_models["sh0"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"])
            else:
                esti_bits = self.entropy_models["sh0"](fq_out_dict["output_value"], fq_out_dict["q_step"])
            return fq_out_dict["output_value"].unsqueeze(1), esti_bits
        else:
            return fq_out_dict["output_value"], None
    

    def simulate_compression_shN(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor) -> Tensor:
        if self.shN_ada_mask_opt and step > self.shN_ada_mask_step:
            param = self.shN_ada_mask(param, step)
            return param, None

        return param, None


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