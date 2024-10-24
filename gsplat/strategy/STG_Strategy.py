from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split, _update_param_with_optimizer
from typing_extensions import Literal


@dataclass
class STG_Strategy(Strategy):
    '''A Densification and Pruning strategy that follows the spacetime gaussian paper:
    
    '''
    # TODO check if the following params are still needed
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15 # default not used
    refine_scale2d_stop_iter: int = 0 # default not used
    refine_start_iter: int = 500
    refine_stop_iter: int = 9_000 # 15_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    # key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale} # scene_scale = 距离场景中心最远的相机位置 - 场景中心位置
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities", "trbf_scale", "trbf_center", "motion", "omega"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            "means2d" in info
        ), "The 2D means of the Gaussians is required but missing."
        info["means2d"].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        flag: int, 
        desicnt: int, 
        maxbounds: float, 
        minbounds: float,  
        packed: bool = False,
    ):  
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            # freeze weights of omega
            params["omega"].grad = params["omega"].grad * self.omegamask # TODO check if this is proceed as expected
            self.rotationmask = torch.logical_not(self.omegamask)
            # freeze weights of rotation
            params["quats"].grad = params["quats"].grad * self.rotationmask # TODO check if this is proceed as expected
            if step % 1000 == 500 :
                zmask = params["means"][:,2] < 4.5  
                remove(params=params, optimizers=optimizers, state=state, mask=zmask)
                self.omegamask = self._zero_omegabymotion(params, optimizers) # calculate omegamask again to adjust the change of gaussian numbers
                torch.cuda.empty_cache()
            if step == 10000: 
                self.removeminmax(params=params, optimizers=optimizers, state=state, maxbounds=maxbounds, minbounds=minbounds)
                self.omegamask = self._zero_omegabymotion(params, optimizers) # calculate omegamask again to adjust the change of gaussian numbers
            return flag 

        self._update_state(params, state, info, packed=packed)
        
        # TODO need to consider more strategy, there are totally 3 types of strategy in STG (densify = 1,2,3)
        # here is a implementation of densify=1
        # omega & rotation mask
        if step ==  8001 :
            omegamask = self._zero_omegabymotion(params, optimizers)
            self.omegamask = omegamask
            # record process
        elif step > 8001: 
            # freeze weights of omega
            params["omega"].grad = params["omega"].grad * self.omegamask # this is likely wrong
            self.rotationmask = torch.logical_not(self.omegamask)
            # freeze weights of rotation
            params["quats"].grad = params["quats"].grad * self.rotationmask # this is likely wrong
        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            if flag < desicnt:
                # grow GSs
                n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(params['means'])} GSs."
                    )
                # according to STG, pruning don't proceed here
                
                # reset running stats
                state["grad2d"].zero_()
                state["count"].zero_()
                if self.refine_scale2d_stop_iter > 0:
                    state["radii"].zero_()
                torch.cuda.empty_cache()
                
                flag+=1
            else:
                if step < 7000 : # defalt 7000. 
                    # prune GSs
                    n_prune = self._prune_gs(params, optimizers, state, step)
                    if self.verbose:
                        print(
                            f"Step {step}: {n_prune} GSs pruned. "
                            f"Now having {len(params['means'])} GSs."
                        )
                    torch.cuda.empty_cache() # check if this is needed
            
        if step % self.reset_every == 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,    
            )   
            
        return flag 

            
        
    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            "means2d",
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel]  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )

        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()
        
        # first duplicate
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        # then split
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            # In STG, scale is not considered when pruning
            # is_too_big = (
            #     torch.exp(params["scales"]).max(dim=-1).values
            #     > self.prune_scale3d * state["scene_scale"]
            # )
            
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            # is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
    
    @torch.no_grad()
    def _zero_omegabymotion(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        threhold=0.15,
        ):
        scales = torch.exp(params["scales"])
        motions = params["motion"]
        pointopacity = torch.sigmoid(params["opacities"])
        omega = params["omega"]
        
        omegamask = torch.sum(torch.abs(motions[:, 0:3]), dim=1) > 0.3
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        opacitymask = pointopacity > 0.7 # Shape here may not be correct
        
        # 1 we keep omega, 0 we freeze omega 
        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask.unsqueeze(1)))
        omeganew = mask.float() * omega
        sel = torch.where(mask)[0]
        
        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "omega":
                return torch.nn.Parameter(omeganew)
            else:
                raise ValueError(f"Unexpected parameter name: {name}")
        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            return v

        # update the parameters and the state in the optimizers
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=["omega"])
        
        return mask
    
    # @torch.no_grad()
    # def _freezweightsbymasknounsqueeze(model, screenlist, mask):
    #     for k in screenlist:
    #         grad_tensor = getattr(getattr(model, k), 'grad') # obtain k.grad from model
    #         newgrad =  mask*grad_tensor #torch.zeros_like(grad_tensor)
    #         setattr(getattr(model, k), 'grad', newgrad)
    #     return  
    
    def logicalorlist(self, listoftensor):
        mask = None 
        for idx, ele in enumerate(listoftensor):
            if idx == 0 :
                mask = ele 
            else:
                mask = torch.logical_or(mask, ele)
        return mask 
    
    @torch.no_grad()
    def removeminmax(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        maxbounds,
        minbounds
        ):
        maxx, maxy, maxz = maxbounds
        minx, miny, minz = minbounds
        xyz = params["means"]
        mask0 = xyz[:,0] > maxx.item()
        mask1 = xyz[:,1] > maxy.item()
        mask2 = xyz[:,2] > maxz.item()

        mask3 = xyz[:,0] < minx.item()
        mask4 = xyz[:,1] < miny.item()
        mask5 = xyz[:,2] < minz.item()
        mask =  self.logicalorlist([mask0, mask1, mask2, mask3, mask4, mask5])
        remove(params=params, optimizers=optimizers, state=state, mask=mask)
        return
    
    
     
