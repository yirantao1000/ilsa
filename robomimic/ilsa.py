from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import robomimic.utils.file_utils as FileUtils




gmm_use_base = True
gmm_means_weight = .5
gmm_scales_weight = .5
gmm_logits_weight = .5

drawer_rotate = True

def find_min_max(task, loc_only, upgraded, latest):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if task == 'pill':
        min_proprios = torch.tensor([])

        max_proprios = torch.tensor([])

        min_actions = torch.tensor([])

        max_actions = torch.tensor([])
    elif task == 'cereal':  
                                   
        min_proprios = torch.tensor([])

        max_proprios = torch.tensor([])

        min_actions = torch.tensor([])

        max_actions = torch.tensor([])
    else:
        raise NotImplementedError
                
    return min_proprios.to(device), max_proprios.to(device), min_actions.to(device), max_actions.to(device)


def Unnormalize(data, min_data, max_data, min=-1, max=1):
    data_std = (data - min) / (max - min)
    original_data = data_std * (max_data - min_data) + min_data   
    return original_data





def manual_sample_base(actions, wrong_possibility = 0.0, xbox_threshold = 0.25, task = None):
    # print(actions.shape)
    # print(actions[0])
   
    actions = actions[:,:3]

    greater_mask = (actions >= 0).float()
    
    sampled_actions = torch.rand_like(actions)
    sampled_actions = greater_mask * sampled_actions + (1 - greater_mask) * (sampled_actions - 1)
    
    
    return sampled_actions


class WrongDirectionLoss(nn.Module):
    def __init__(self, margin=0.0, square = False, task = 'cereal', loc_only = False, upgraded = False, latest = False, xbox_threshold=0.25):
        super(WrongDirectionLoss, self).__init__()
        self.relu = nn.ReLU()
        # self.margin = margin
        if margin != 0:
            raise NotImplementedError
        
        _, _, self.min_actions, self.max_actions = find_min_max(task, loc_only, upgraded, latest)
        
        if square:
            self.exponent = 2
        else:
            self.exponent = 1
        self.xbox_threshold = xbox_threshold


    def forward(self, outputs, user_actions, return_mean = False):
        # if self.limit:
        #     outputs = Unnormalize(outputs[:,:4], self.min_actions[:4], self.max_actions[:4])
        # else:
        #     outputs = Unnormalize(outputs[:,:6], self.min_actions[:6], self.max_actions[:6])
        outputs = Unnormalize(outputs[:,:3], self.min_actions[:3], self.max_actions[:3])
        assert outputs.shape[0]==user_actions.shape[0] and outputs.shape[1]==user_actions.shape[1]==3
    
        positive_mask = (user_actions>=self.xbox_threshold)
        negative_mask = (user_actions<=-self.xbox_threshold)
        
        loss_positive_location = (self.relu(-outputs) * positive_mask.float()) ** self.exponent
        loss_negative_location = (self.relu(outputs) * negative_mask.float()) ** self.exponent
        
        if return_mean:
            loss = (loss_positive_location + loss_negative_location).mean()
        else:
            loss = (loss_positive_location + loss_negative_location).mean(dim=1)
        # print(loss.shape)
        # exit()
        return loss

        

class MarginLossForMajorDirection(nn.Module):
    def __init__(self, margin=0.0, square = False, task = 'cereal', loc_only = False, upgraded = False, latest = False, xbox_threshold=0.25):
        super(MarginLossForMajorDirection, self).__init__()
        # self.margin = margin
        if margin != 0:
            raise NotImplementedError
        _, _, self.min_actions, self.max_actions = max_actions = find_min_max(task, loc_only, upgraded, latest)
        if square:
            self.exponent = 2
        else:
            self.exponent = 1
        self.xbox_threshold = xbox_threshold

    def forward(self, outputs, user_actions, return_mean = False):
        
        outputs = Unnormalize(outputs[:,:3], self.min_actions[:3], self.max_actions[:3])  
        assert outputs.shape[0]==user_actions.shape[0] and outputs.shape[1]==user_actions.shape[1]==3
        
        if len(outputs.shape)==1:
            raise NotImplementedError
            user_actions = torch.tensor([user_actions])
            outputs = outputs.unsqueeze(0)
       
        
        abs_outputs = torch.abs(outputs)
        abs_user_actions = torch.abs(user_actions)
        
        # Expanding user_actions to compare every element with every other element
        user_actions_expanded = abs_user_actions.unsqueeze(2)  # Shape: (batch_size, 3, 1)
        user_actions_t = abs_user_actions.unsqueeze(1)         # Shape: (batch_size, 1, 3)
        
        # Calculate the condition where one is greater than the other and greater than 0.25
        greater = user_actions_expanded > user_actions_t
        significant = user_actions_expanded > self.xbox_threshold
        
        valid_conditions = greater & significant
        
        # Expanding outputs in the same way to enable element-wise comparisons
        outputs_expanded = abs_outputs.unsqueeze(2)
        outputs_t = abs_outputs.unsqueeze(1)
        
        # Calculate the ReLU loss only where valid_conditions is True
        if return_mean:
            losses = ((F.relu(outputs_t - outputs_expanded) * valid_conditions.float()) ** self.exponent).mean()
        else:
            losses = ((F.relu(outputs_t - outputs_expanded) * valid_conditions.float()) ** self.exponent).mean(dim=(1,2))
        # print(losses.shape)
        # exit()      
        
        return losses







@register_algo_factory_func("ilsa")
def algo_config_to_class(algo_config):   
    return ILSA_Transformer_Finish, {}
    
   
        



class ILSA(PolicyAlgo):

    def train_on_batch(self, batch, epoch, validate=False):
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(ILSA, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(ILSA, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()

        if "wrong_direction_loss" in info["losses"]:
            log["wrong_direction_loss"] = info["losses"]["wrong_direction_loss"].item()
        if "squared_wrong_direction_loss" in info["losses"]:
            log["squared_wrong_direction_loss"] = info["losses"]["squared_wrong_direction_loss"].item()
        if "major_direction_loss" in info["losses"]:
            log["major_direction_loss"] = info["losses"]["major_direction_loss"].item()
        if "squared_major_direction_loss" in info["losses"]:
            log["squared_major_direction_loss"] = info["losses"]["squared_major_direction_loss"].item()
        
        if "gt_loss" in info["losses"]:
            log["gt_loss"] = info["losses"]["gt_loss"].item()
        if "user_loss" in info["losses"]:
            log["user_loss"] = info["losses"]["user_loss"].item()

        if "middle_gt_loss" in info["losses"]:
            log["middle_gt_loss"] = info["losses"]["middle_gt_loss"].item()

        if "finish_loss" in info["losses"]:
            log["finish_loss"] = info["losses"]["finish_loss"].item()


        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class ILSA_Transformer(ILSA):
    def _create_networks(self):
        self.nets = nn.ModuleDict()

        assert self.ac_dim == 4
        self.user_actions_shape = 3

        # self.obs_shapes["user_actions"] = 3
       
        self.nets["policy"] = PolicyNets.TransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            end2end = self.algo_config.end2end,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )

        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)
        # print(self.nets)
        # exit()
        
    
    def _set_params_from_config(self):
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps

    def process_batch_for_training(self, batch):
        input_batch = dict()
        h = self.context_length 
        # print(h) #10
        # print(batch["obs"]['proprios'].shape) #torch.Size([100, 20, 13])
        # print(batch["obs"]['proprios'][0,:,0:2])
        # print(batch["actions"].shape) #torch.Size([100, 20, 6])
        # print(batch['actions'][0,:,0:2]) 
        # exit()
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        if self.supervise_all_steps:
            raise NotImplemented
            # supervision on entire sequence (instead of just current timestep)
            input_batch["actions"] = batch["actions"][:, :h, :]
        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h-1, :]
            if "user_actions" in batch.keys():
                input_batch["user_actions"] = batch["user_actions"][:, :h]
            if "modified_actions" in batch.keys():
                input_batch["modified_actions"] = batch["modified_actions"][:, h-1,:]
                # print(batch["modified_actions"])
                # exit()
            if "corrects" in batch.keys():
                input_batch["corrects"] = batch["corrects"][:, h-1]

            if "modified" in batch.keys():
                input_batch["modified"] = batch["modified"][:, h-1]
            
            if "inits" in batch.keys():
                input_batch["inits"] = batch["inits"][:, h-1]

            if "intervention" in batch.keys():
                input_batch["intervention"] = batch["intervention"][:, h-1]

            if "pre_intervention" in batch.keys():
                input_batch["pre_intervention"] = batch["pre_intervention"][:, h-1]

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch

    def _forward_training(self, batch, epoch=None):
        # print(batch['obs']['proprios'].shape)
        # print(batch['obs']['proprios'][8,0])
        # print(batch['obs']['proprios'][8,1])
        # exit()
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        predictions = OrderedDict()

        if self.algo_config.incremental:
            raise NotImplementedError
            user_actions = batch["user_actions"]
            indices = indices.long()
            assert len(indices.shape)==2 and indices.shape[1]==self.context_length
            batch["obs"]["base_classifications"] = torch.empty(batch["obs"]["objects"].shape[0], self.context_length, self.algo_config.classification_input_num)
            # print(indices)
            # print(indices.shape)
            # exit()
            if self.algo_config.actual_action_base:
                raise NotImplementedError
            else:
                assert self.algo_config.one_hot_base
                for t in range(self.context_length):
                    batch["obs"]["base_classifications"][:,t,:] = F.one_hot(indices[:,t], num_classes=self.algo_config.classification_input_num).float()
        
        else:
            # batch["obs"]["user_actions"] = manual_sample_base(batch["obs"]["proprios"][:,-1,:], batch["obs"]["objects"][:,-1,:], upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)
            assert len(batch["actions"].shape)==2
            batch["obs"]["user_actions"] = manual_sample_base(batch["actions"])
            
                    
        predictions["middle_actions"], predictions["actions"] = self.nets["policy"](obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"])
        if not self.supervise_all_steps:
            # only supervise final timestep
            # predictions["actions"] = predictions["actions"][:, -1, :]
            predictions["middle_actions"] = predictions["middle_actions"][:, -1, :]

        #prepare for loss calculation
        if not self.algo_config.incremental:           
            if self.supervise_all_steps:   
                raise NotImplementedError
                predictions["user_indices"] =  []
                for t in range(self.context_length):
                    if self.algo_config.actual_action_base:
                        predictions["user_indices"].append(actions2indices(batch["obs"]["base_actions"][:,t,:], normalized=self.algo_config.normalized_action_base))    
                    elif self.algo_config.one_hot_base:
                        predictions["user_indices"].append(batch["obs"]["base_classifications"][:,t,:].max(dim=1)[1])
            else:    
                # predictions["user_actions"] = batch["obs"]["user_actions"][:,-1,:]
                # print(batch['obs'].keys())
                # exit()
                predictions["user_actions"] = batch["obs"]["user_actions"]
                       
        return predictions
    
    def _compute_losses(self, predictions, batch):
        losses = OrderedDict()
        a_target = batch["actions"]
        middle_actions = predictions["middle_actions"]
        actions = predictions["actions"]
        if a_target.shape[-1]==actions.shape[-1]+1:
            a_target = a_target[:,:-1]

        if self.algo_config.incremental and self.algo_config.sample_modified_action:
            raise NotImplementedError
            modified_actions = batch["modified_actions"]
            a_target = torch.where((torch.rand(a_target.shape[0], 1) < self.algo_config.modified_ratio).expand(-1, 6), modified_actions, a_target)

        if self.algo_config.incremental:
            raise NotImplementedError
            indices = batch["user_actions"][:,-1].long()
            corrects = batch["corrects"]
            assert len(indices.shape)==len(corrects.shape)==1
            if self.algo_config.distill_on_wrong:
                gt_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool)
            else:
                gt_mask = corrects.squeeze().bool()
                      
            user_mask = ~ (corrects.squeeze().bool())
            if self.algo_config.filter_wrong:
                raise NotADirectoryError
                cols = torch.arange(a_target.shape[1]).expand(a_target.shape[0], -1)
                wrong_action_dim_mask = cols == (indices//2).unsqueeze(1)
                wrong_mask = ~corrects.to(torch.bool).unsqueeze(1).expand_as(actions)
                final_mask = wrong_action_dim_mask & wrong_mask
                # actions[final_mask] = 0
                # a_target[final_mask] = 0
                a_target[final_mask] = actions[final_mask]


        else:
            user_actions = predictions["user_actions"]
            gt_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool)
            user_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool)


    
        losses["l2_loss"] = nn.MSELoss()(actions[gt_mask], a_target[gt_mask])
        losses["middle_l2_loss"] = nn.MSELoss()(middle_actions[gt_mask], a_target[gt_mask])

        losses["middle_gt_loss"] = self.algo_config.loss.l2_weight * losses["middle_l2_loss"]
        losses["gt_loss"] = self.algo_config.loss.l2_weight * losses["l2_loss"]
        
       

        if torch.sum(user_mask) > 0:

            if self.supervise_all_steps:  
                return NotImplementedError 
                
            else:
                losses["wrong_direction_loss"] = WrongDirectionLoss(square=False, limit = self.algo_config.limit, location_only = self.algo_config.location_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask])
                losses["major_direction_loss"] = MarginLossForMajorDirection(square=False, limit = self.algo_config.limit, location_only = self.algo_config.location_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask])

                losses["squared_wrong_direction_loss"] = WrongDirectionLoss(square=True, limit = self.algo_config.limit, location_only = self.algo_config.location_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask])
                losses["squared_major_direction_loss"] = MarginLossForMajorDirection(square=True, limit = self.algo_config.limit, location_only = self.algo_config.location_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask])
                
                

            user_losses = [
                self.algo_config.wrong_direction_loss * losses["wrong_direction_loss"],
                self.algo_config.major_direction_loss * losses["major_direction_loss"],

                self.algo_config.squared_wrong_direction_loss * losses["squared_wrong_direction_loss"],
                self.algo_config.squared_major_direction_loss * losses["squared_major_direction_loss"],
                
            ]

            losses["user_loss"] = sum(user_losses)
            losses["action_loss"] = losses["gt_loss"] * self.algo_config.gt_weight + losses["user_loss"] * self.algo_config.user_weight + losses["middle_gt_loss"] * self.algo_config.middle_gt_weight
    
        else:
            losses["action_loss"] = losses["gt_loss"] * self.algo_config.gt_weight + losses["middle_gt_loss"] * self.algo_config.middle_gt_weight
    
        return losses
    

    def get_action(self, obs_dict, goal_dict=None):
        if len(obs_dict['proprios'].shape)==2:
            obs_dict['proprios'] = obs_dict['proprios'].unsqueeze(0)
            obs_dict['objects'] = obs_dict['objects'].unsqueeze(0)
        # print(obs_dict['proprios'].shape) #torch.Size([1, 1, 13])
        # exit()
        assert not self.nets.training
        outputs = self.nets["policy"](obs_dict, actions=None, goal_dict=goal_dict)
        return outputs[0][:, -1, :], outputs[1]





class ILSA_Transformer_Finish(ILSA_Transformer):
    def _create_networks(self):
        self.nets = nn.ModuleDict()
        
        assert self.algo_config.finish
        assert self.ac_dim == 6
            
        self.user_actions_shape = 3

        # self.obs_shapes["user_actions"] = 3
        if self.algo_config.finish_separate:
            self.nets["policy"] = PolicyNets.TransformerFinishNetwork(
                obs_shapes=self.obs_shapes,
                goal_shapes=self.goal_shapes,
                ac_dim=self.ac_dim,
                mlp_layer_dims=self.algo_config.actor_layer_dims,
                finish_layer_dims = self.algo_config.finish_layer_dims,
                end2end = self.algo_config.end2end,
                encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
                **BaseNets.transformer_args_from_config(self.algo_config.transformer),
            )
        else:
            raise NotImplementedError
           

        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch, epoch=None):
        # print(batch['obs']['proprios'].shape)
        # print(batch['obs']['proprios'][8,0])
        # print(batch['obs']['proprios'][8,1])
        # exit()
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        predictions = OrderedDict()

        if self.algo_config.incremental:
            # print(batch["user_actions"].shape) #torch.Size([100, 5, 4])
            # exit()
            batch["user_actions"] = batch["user_actions"][:,-1,:]
            if batch["user_actions"].shape[-1]==4:
                batch["user_actions"] = batch["user_actions"][:,:-1]
                # print(batch["user_actions"])
                # exit()
            if self.algo_config.manual_sample_incre:
                batch["user_actions"] = manual_sample_base(batch["actions"], task = self.algo_config.task)
                # batch["user_actions"] = torch.where(batch["corrects"].unsqueeze(1) == 0, batch["user_actions"], manual_sample_base(batch["actions"]))
            else:
                batch["user_actions"] = batch["user_actions"]

            # elif self.algo_config.manual_sample_modified:
            #     # batch["user_actions"] = torch.where(batch["modified"].unsqueeze(1) == 0, batch["user_actions"], manual_sample_base(batch["obs"]["proprios"][:,-1,:], batch["obs"]["objects"][:,-1,:], upgraded = self.algo_config.upgraded, latest = self.algo_config.latest))
            #     batch["user_actions"] = torch.where(batch["modified"].unsqueeze(1) == 0, batch["user_actions"], manual_sample_base(batch["actions"]))
            
            batch["obs"]["user_actions"] = batch["user_actions"]
            
        else:
            assert len(batch["actions"].shape) == 2
            batch["obs"]["user_actions"] = manual_sample_base(batch["actions"])
            
        if self.algo_config.finish_separate:
            predictions["finish"], predictions["middle_actions"], predictions["actions"] = self.nets["policy"](obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"])
        else:  
            predictions["middle_actions"], predictions["actions"] = self.nets["policy"](obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"])
        if not self.supervise_all_steps:
            # only supervise final timestep
            # predictions["actions"] = predictions["actions"][:, -1, :]
            predictions["middle_actions"] = predictions["middle_actions"][:, -1, :]

        #prepare for loss calculation
        # if not self.algo_config.incremental:           
        #     if self.supervise_all_steps:   
        #         raise NotImplementedError
        #         predictions["user_indices"] =  []
        #         for t in range(self.context_length):
        #             if self.algo_config.actual_action_base:
        #                 predictions["user_indices"].append(actions2indices(batch["obs"]["base_actions"][:,t,:], normalized=self.algo_config.normalized_action_base))    
        #             elif self.algo_config.one_hot_base:
        #                 predictions["user_indices"].append(batch["obs"]["base_classifications"][:,t,:].max(dim=1)[1])
        #     else:    
        #         # predictions["user_actions"] = batch["obs"]["user_actions"][:,-1,:]
        #         # print(batch['obs'].keys())
        #         # exit()
        #         predictions["user_actions"] = batch["obs"]["user_actions"]
                       
        return predictions
    


    def _compute_losses(self, predictions, batch):
        # if self.algo_config.incre_transformer_only:
        #     assert self.algo_config.incre_transformer_only
        #     self.algo_config.gt_weight = self.algo_config.user_weight = 0
        losses = OrderedDict()
        a_target = batch["actions"]
        middle_actions = predictions["middle_actions"]
        actions = predictions["actions"]
        user_actions = batch["obs"]["user_actions"]
        if a_target.shape[-1]==actions.shape[-1]+1:
            a_target = a_target[:,:-1]

    
        if self.algo_config.incremental:

            modified = batch["modified"]
            corrects = batch["corrects"]
            inits = batch["inits"]

            assert len(modified.shape)==len(corrects.shape)==len(inits.shape)==1
            

            if self.algo_config.middle_gt_all:
                middle_gt_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool).to('cuda:0')
            elif self.algo_config.no_middle_gt:
                middle_gt_mask = torch.zeros(batch["actions"].shape[0], dtype=torch.bool).to('cuda:0')
            else:
                middle_gt_mask = inits.squeeze().bool()

            if not self.algo_config.middle_gt_wrong:
                middle_gt_mask = middle_gt_mask & (corrects.squeeze().bool())



            if self.algo_config.final_gt_all:
                gt_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool).to('cuda:0')
            else:
                gt_mask = ~(inits.squeeze().bool())

            if not self.algo_config.final_gt_wrong:
                gt_mask = gt_mask & (corrects.squeeze().bool())
                
            finish_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool)
            assert not self.algo_config.wrong_user_loss
            user_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool)



        else:
            gt_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool)
            middle_gt_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool)
            user_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool)
            finish_mask = torch.ones(batch["actions"].shape[0], dtype=torch.bool)

   
        if not (self.algo_config.reweight or self.algo_config.fancy_reweight):
            # exit()
            losses["l2_loss"] = nn.MSELoss()(actions[gt_mask], a_target[gt_mask])
            losses["middle_l2_loss"] = nn.MSELoss()(middle_actions[middle_gt_mask], a_target[middle_gt_mask])
            if self.algo_config.middle_gt_weight > 0 and not self.algo_config.no_middle_gt:
                losses["middle_gt_loss"] = self.algo_config.loss.l2_weight * losses["middle_l2_loss"]
            losses["gt_loss"] = self.algo_config.loss.l2_weight * losses["l2_loss"]
            
            if self.algo_config.middle_gt_weight > 0 and not self.algo_config.no_middle_gt:
                losses["action_loss"] = losses["gt_loss"] * self.algo_config.gt_weight + losses["middle_gt_loss"] * self.algo_config.middle_gt_weight
            else:
                losses["action_loss"] = losses["gt_loss"] * self.algo_config.gt_weight

            if torch.sum(user_mask) > 0:
                if self.supervise_all_steps:  
                    return NotImplementedError 
                else:
                    losses["wrong_direction_loss"] = WrongDirectionLoss(square=False, task = self.algo_config.task, loc_only = self.algo_config.loc_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask], return_mean = True)
                    losses["major_direction_loss"] = MarginLossForMajorDirection(square=False, task = self.algo_config.task, loc_only = self.algo_config.loc_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask], return_mean = True)

                    losses["squared_wrong_direction_loss"] = WrongDirectionLoss(square=True, task = self.algo_config.task, loc_only = self.algo_config.loc_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask], return_mean = True)
                    losses["squared_major_direction_loss"] = MarginLossForMajorDirection(square=True, task = self.algo_config.task, loc_only = self.algo_config.loc_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask], return_mean = True)
                    
                    
                user_losses = [
                    self.algo_config.wrong_direction_loss * losses["wrong_direction_loss"],
                    self.algo_config.major_direction_loss * losses["major_direction_loss"],

                    self.algo_config.squared_wrong_direction_loss * losses["squared_wrong_direction_loss"],
                    self.algo_config.squared_major_direction_loss * losses["squared_major_direction_loss"],
                    
                ]

                losses["user_loss"] = sum(user_losses)
                losses["action_loss"] += losses["user_loss"] * self.algo_config.user_weight
        
            if self.algo_config.finish_separate:
                losses["finish_loss"] = nn.MSELoss()(predictions["finish"][finish_mask], a_target[:,-1].unsqueeze(1)[finish_mask])   
                losses["action_loss"] += losses["finish_loss"] * self.algo_config.finish_weight 


        else:
            losses["l2_loss"] = (nn.MSELoss(reduction='none')(actions[gt_mask], a_target[gt_mask])).mean(dim=1)
            losses["middle_l2_loss"] = (nn.MSELoss(reduction='none')(middle_actions[middle_gt_mask], a_target[middle_gt_mask])).mean(dim=1)
            if self.algo_config.middle_gt_weight > 0 and not self.algo_config.no_middle_gt:
                losses["middle_gt_loss"] = self.algo_config.loss.l2_weight * losses["middle_l2_loss"]
            losses["gt_loss"] = self.algo_config.loss.l2_weight * losses["l2_loss"]
            
            if self.algo_config.middle_gt_weight > 0 and not self.algo_config.no_middle_gt:
                losses["action_loss"] = losses["gt_loss"] * self.algo_config.gt_weight + losses["middle_gt_loss"] * self.algo_config.middle_gt_weight
            else:
                losses["action_loss"] = losses["gt_loss"] * self.algo_config.gt_weight

            if torch.sum(user_mask) > 0:
                if self.supervise_all_steps:  
                    return NotImplementedError 
                else:
                    losses["wrong_direction_loss"] = WrongDirectionLoss(square=False, task = self.algo_config.task, loc_only = self.algo_config.loc_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask])
                    losses["major_direction_loss"] = MarginLossForMajorDirection(square=False, task = self.algo_config.task, loc_only = self.algo_config.loc_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask])

                    losses["squared_wrong_direction_loss"] = WrongDirectionLoss(square=True, task = self.algo_config.task, loc_only = self.algo_config.loc_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask])
                    losses["squared_major_direction_loss"] = MarginLossForMajorDirection(square=True, task = self.algo_config.task, loc_only = self.algo_config.loc_only, upgraded = self.algo_config.upgraded, latest = self.algo_config.latest)(actions[user_mask], user_actions[user_mask])
                    
                    
                user_losses = [
                    self.algo_config.wrong_direction_loss * losses["wrong_direction_loss"],
                    self.algo_config.major_direction_loss * losses["major_direction_loss"],

                    self.algo_config.squared_wrong_direction_loss * losses["squared_wrong_direction_loss"],
                    self.algo_config.squared_major_direction_loss * losses["squared_major_direction_loss"],
                    
                ]

                losses["user_loss"] = sum(user_losses)
                losses["action_loss"] += losses["user_loss"] * self.algo_config.user_weight
        
            if self.algo_config.finish_separate:
                losses["finish_loss"] = (nn.MSELoss(reduction='none')(predictions["finish"][finish_mask], a_target[:,-1].unsqueeze(1)[finish_mask])).mean(dim=1)   
                losses["action_loss"] += losses["finish_loss"] * self.algo_config.finish_weight     

            for key, loss in losses.items():
                inits = inits.bool()
                inits_num = torch.sum(inits).item()
                all_num = inits.shape[0]

                if self.algo_config.fancy_reweight:
                    intervention = batch["intervention"].int()
                    pre_intervention = batch["pre_intervention"].int()
                    inits_int = inits.int()
                    on_policy = (~inits & ~intervention & ~pre_intervention).int()
                    intervention_num = torch.sum(intervention).item()
                    pre_intervention_num = torch.sum(pre_intervention).item()
                    on_policy_num = torch.sum(on_policy).item()
                    new_num = torch.sum(~inits).item()
                    assert intervention_num + pre_intervention_num + on_policy_num == new_num
                    if intervention_num != 0:
                        intervention_weight = all_num/intervention_num * 0.5
                    else:
                        intervention_weight = 0
                    if on_policy_num !=0:   
                        on_policy_weight = all_num/on_policy_num * 0.25
                    else:
                        on_policy_weight = 0
                    assert inits_num !=0
                    inits_weight = all_num/inits_num * 0.25

                    weighted_loss = loss * ((inits * inits_weight) + (intervention * intervention_weight) + (on_policy * on_policy_weight))
                    losses[key] = weighted_loss.mean()
                else:
                    assert self.algo_config.reweight
                    new = (~inits).int()
                    new_num = torch.sum(new).item()
                    # print(inits_num)
                    # print(new)
                    # print(new_num)
                    # exit()
                    if new_num > 0:
                        new_weight = all_num/new_num * 0.5
                    else:
                        new_weight = 0
                    inits_weight = all_num/inits_num * 0.5
                    

                    weighted_loss = loss * ((inits * inits_weight) + (new * new_weight))
                    losses[key] = weighted_loss.mean()
        
                

        return losses


    def get_action(self, obs_dict, goal_dict=None):
        if len(obs_dict['proprios'].shape)==2:
            obs_dict['proprios'] = obs_dict['proprios'].unsqueeze(0)
            obs_dict['objects'] = obs_dict['objects'].unsqueeze(0)
        # print(obs_dict['proprios'].shape) #torch.Size([1, 1, 13])
        # exit()
        assert not self.nets.training
        outputs = self.nets["policy"](obs_dict, actions=None, goal_dict=goal_dict)

        if self.algo_config.finish_separate:       
            return outputs[0], outputs[1][:, -1, :], outputs[2]
        else:
            return outputs[0][:, -1, :], outputs[1]

