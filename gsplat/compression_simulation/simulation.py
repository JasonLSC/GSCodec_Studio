from dataclasses import dataclass
from hmac import new
from typing import Dict, Literal, Optional, Tuple, Union, Callable

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
                elif isinstance(v, Entropy_gaussian):
                    # TODO
                    pass 
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
            if isinstance(self.entropy_models["quats"], Entropy_gaussian):
                pass
                
            elif isinstance(self.entropy_models["quats"], Entropy_factorized_optimized_refactor):
                if choose_idx is not None:
                    esti_bits = self.entropy_models["quats"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"])
                else:
                    esti_bits = self.entropy_models["quats"](fq_out_dict["output_value"], fq_out_dict["q_step"])

            else:
                raise NotImplementedError

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


class STGCompressionSimulation:
    """
    """
    def __init__(self, quantization_sim_type: Optional[Literal["round", "noise", "vq"]] = None,
                 entropy_model_enable: bool = False,
                 entropy_steps: Dict[str, int] = None, 
                 device: device = None, 
                 ada_mask_opt: bool = False,
                 ada_mask_step: int = 10_000,
                 **kwargs) -> None:
        self.quantization_sim_type = quantization_sim_type

        self.entropy_model_enable = entropy_model_enable
        self.entropy_steps = entropy_steps
        self.device = device

        # TODO: 明确写一个函数，根据splats里有哪些元素，以及配置文件中让哪些元素参与simulation

        # simulation_option: dict to specify which properties should be involved in the compression simulation.
        # Once option is set to True, it must have corresponding simulate_fn
        self.simulation_option = {
            "means": False,
            "scales": True,
            "quats": True,
            "opacities": True,
            "trbf_center": False,
            "trbf_scale": False,
            "motion": False, # [N, 9]
            "omega": False, # [N, 4]
            "colors": True,
            "features_dir": True,
            "features_time": True,
        }

        self.shN_qat = False
        self.shN_ada_mask_opt = ada_mask_opt
        self.shN_ada_mask_step = ada_mask_step

        # configs for "differentiable quantization"
        self.q_bitwidth = {
            "means": None,
            "scales": 8,
            "quats": 8,
            "opacities": 8,
            "trbf_center": None,
            "trbf_scale": None,
            "motion": None, # [N, 9]
            "omega": None, # [N, 4]
            "colors": 8,
            "features_dir": 8,
            "features_time": 8,
        }

        self.bds = {
            "means": None,
            "scales": [-10, 2],
            "quats": [-1, 1],
            "opacities": [-7, 7],
            "trbf_center": None,
            "trbf_scale": None,
            "motion": None, # [N, 9]
            "omega": None, # [N, 4]
            "colors": [-7.5, 7.5],
            "features_dir": [-10, 10],
            "features_time": [-10, 10],
        }

        # configs for "entropy constraint"
        self.entropy_model_option = {
            "means": False,
            "scales": True,
            "quats": True,
            "opacities": False,
            "colors": True,
            "features_dir": True,
            "features_time": True
            # "shN": False
        }

        if self.entropy_model_enable:
            self.entropy_models = {
                "means": None,
                "scales": Entropy_factorized_optimized_refactor(channel=3).to(self.device),
                # "scales": None,
                "quats": Entropy_factorized_optimized_refactor(channel=4).to(self.device),
                "opacities": None,
                "colors": Entropy_factorized_optimized_refactor(channel=3, filters=(3, 3)).to(self.device),
                "features_dir": Entropy_factorized_optimized_refactor(channel=3, filters=(3, 3)).to(self.device),
                "features_time": Entropy_factorized_optimized_refactor(channel=3, filters=(3, 3)).to(self.device),
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

        # configs for "adaptive mask"
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
            # "trbf_center": self.simulate_compression_trbf_center,
            # "trbf_scale": self.simulate_compression_trbf_scale,
            # "motion": self.simulate_compression_motion,
            # "omega": self.simulate_compression_omega,
            "colors": self.simulate_compression_colors,
            "features_dir": self.simulate_compression_features_dir,
            "features_time": self.simulate_compression_features_time
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
            fq_out_dict = fake_quantize_ste(param, self.bds["quats"][0], self.bds["quats"][1], 8, self.quantization_sim_type)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["quats"][0], self.bds["quats"][1], self.q_bitwidth["quats"], self.quantization_sim_type)
        
        # entropy constraint
        if step > self.entropy_steps["quats"] and self.entropy_model_enable and self.entropy_model_option["quats"]:
            # import pdb; pdb.set_trace()
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
            fq_out_dict = fake_quantize_ste(param, self.bds["scales"][0], self.bds["scales"][1], 8, self.quantization_sim_type)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["scales"][0], self.bds["scales"][1], self.q_bitwidth["scales"], self.quantization_sim_type)

        # entropy constraint
        if step > self.entropy_steps["scales"] and self.entropy_model_enable and self.entropy_model_option["scales"]:
            # import pdb; pdb.set_trace()
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
        fq_out_dict = fake_quantize_ste(param, self.bds["opacities"][0], self.bds["opacities"][1], 8, self.quantization_sim_type)

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


    def simulate_compression_colors(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor) -> Tensor:
        # fake quantize
        if step < 10_000:
            fq_out_dict = fake_quantize_ste(param, self.bds["colors"][0], self.bds["colors"][1], 8, self.quantization_sim_type)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["colors"][0], self.bds["colors"][1], self.q_bitwidth["colors"], self.quantization_sim_type)
        
        # entropy constraint
        if step > self.entropy_steps["colors"] and self.entropy_model_enable and self.entropy_model_option["colors"]:
            fq_out_dict["output_value"] = fq_out_dict["output_value"]
            if choose_idx is not None:
                esti_bits = self.entropy_models["colors"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"])
            else:
                esti_bits = self.entropy_models["colors"](fq_out_dict["output_value"], fq_out_dict["q_step"])
            return fq_out_dict["output_value"], esti_bits
        else:
            return fq_out_dict["output_value"], None
    

    def simulate_compression_features_dir(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor) -> Tensor:
        # fake quantize
        if step < 10_000:
            fq_out_dict = fake_quantize_ste(param, self.bds["features_dir"][0], self.bds["features_dir"][1], 8, self.quantization_sim_type)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["features_dir"][0], self.bds["features_dir"][1], self.q_bitwidth["features_dir"], self.quantization_sim_type)

        # entropy constraint
        if step > self.entropy_steps["features_dir"] and self.entropy_model_enable and self.entropy_model_option["features_dir"]:
            fq_out_dict["output_value"] = fq_out_dict["output_value"]
            if choose_idx is not None:
                esti_bits = self.entropy_models["features_dir"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"])
            else:
                esti_bits = self.entropy_models["features_dir"](fq_out_dict["output_value"], fq_out_dict["q_step"])
            return fq_out_dict["output_value"], esti_bits
        else:
            return fq_out_dict["output_value"], None
    

    def simulate_compression_features_time(self, param: torch.nn.Parameter, step: int, choose_idx: torch.Tensor) -> Tensor:
        # fake quantize
        if step < 10_000:
            fq_out_dict = fake_quantize_ste(param, self.bds["features_time"][0], self.bds["features_time"][1], 8, self.quantization_sim_type)
        else:
            fq_out_dict = fake_quantize_ste(param, self.bds["features_time"][0], self.bds["features_time"][1], self.q_bitwidth["features_time"], self.quantization_sim_type)

        # entropy constraint
        if step > self.entropy_steps["features_time"] and self.entropy_model_enable and self.entropy_model_option["features_time"]:
            fq_out_dict["output_value"] = fq_out_dict["output_value"]
            if choose_idx is not None:
                esti_bits = self.entropy_models["features_time"](fq_out_dict["output_value"][choose_idx], fq_out_dict["q_step"])
            else:
                esti_bits = self.entropy_models["features_time"](fq_out_dict["output_value"], fq_out_dict["q_step"])
            return fq_out_dict["output_value"], esti_bits
        else:
            return fq_out_dict["output_value"], None