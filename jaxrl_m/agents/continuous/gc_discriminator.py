import copy
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax

from jaxrl_m.agents.continuous.iql import iql_value_loss
from jaxrl_m.agents.continuous.iql import iql_actor_loss
from jaxrl_m.agents.continuous.iql import iql_critic_loss
from jaxrl_m.agents.continuous.iql import expectile_loss
from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import GCEncodingWrapper
from jaxrl_m.networks.actor_critic_nets import ValueCritic
from jaxrl_m.networks.mlp import MLP


class GCDiscriminatorAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        batch_size = batch["terminals"].shape[0]
        # neg_goal_indices = jnp.roll(jnp.arange(batch_size, dtype=jnp.int32), -1)

        # # selects a portion of goals to make negative
        # def get_goals_rewards(key):
        #     neg_goal_mask = (
        #         jax.random.uniform(key, (batch_size,))
        #         < self.config["negative_proportion"]
        #     )
        #     goal_indices = jnp.where(
        #         neg_goal_mask, neg_goal_indices, jnp.arange(batch_size)
        #     )
        #     new_goals = jax.tree_map(lambda x: x[goal_indices], batch["goals"])
        #     new_rewards = jnp.where(neg_goal_mask, -1, batch["rewards"])
        #     return new_goals, new_rewards
    

        # def value_loss_fn(params, rng):
        #     rng, key = jax.random.split(rng)
        #     goals, _ = get_goals_rewards(key)

        #     rng, key = jax.random.split(rng)
        #     v = self.state.apply_fn(
        #         {"params": params},  # gradient flows through here
        #         (batch["observations"], goals),
        #         train=True,
        #         rngs={"dropout": key},
        #         name="value",
        #     )
        #     return iql_value_loss(q, v, self.config["expectile"])
        

        def value_loss_fn(params, rng):
            rng, key = jax.random.split(rng)

            batch_size = batch["observations"]["image"].shape[0]

            


            # 4 options: adjacent obs, encode decode, generated 
            # adjacent obsd doesn't make sense for a discriminator? 
                # does it even make sense for this formulation of a contrastive value function? 
            
            # For this discriminator set up, the language prompt isn't used at all? Should we get rid of it? 
            
            # Here a contrastive Vf is where we switch up the language prompts? That's the part we are contrasting? 
            
            # Could do a timestep distance to end of episode style Vf? 

            # Should the discriminator/Vf also get the current obs? To enforce temporal consistency? 

        
            # in the language conditioned setting, a contrastive Vf isn't meaningfully different from a discriminator/classifier 
                # well, basically just a classifier of whether or not an image matches a given language instruction


            # Possible version of the discriminator: gets current obs and goal image, discriminates if it is a true or generated goal image
                # positives: real goal images, encode-decode goal images (and adjacent real and encode-decode goal images?)
                    # maybe just make the goal_kwargs something like (16, 24) instead of (0, 24)
                # negatives: generated goal images
                    # maybe not necessary to do adjacent stuff here? 
            # in this version, adjacent augmentations make sense 


            # ================= # 

            # Are there any existing language conditioned Vf implementations that we could just use? 

            # Formulations that make sense 
            # discriminator w/ current image and goal image
                # goal image during training: real goal image, encode_decode goal image, generated goal image
                    # for real and encode_decode, 
                # having both the current iumage obs and the goal image will allow it to do things like learn temporal consistency, etc. 
                # Might be useful to perform an ablation where I just zero out the current image obs, to see how much it needs the current image obs 
                # in order to determine temporal consistency, etc.  
            
            # Vf: just one image + the language instruction 
                # Could do HIQL Vf formulation, and enforce the value of the encode_decode images to be the same, while pushing down the values of generated images 
                # Could do a contrastive formulation: 
                    # positives: real images and encode_decode images + correct language goal. 
                    # Negatives: generated images + incorrect language goal. real images, encode_decode images, and generated images + incorrect language goal. 
                        # For the positives, would all of the images in a trajectory for a matching language instruction be positives, or just 
                        # the final image (or N images)? 
                            # This is essentially just a classifier of whether or not a given image is fulfulling a given language instruction? 
                                # Potential problem: Wouldn't learn to give higher value to ending images that are actually close to completing the instruction vs. earlier images that aren't close to completing the instruction?
                                    # Maybe this is okay? 
                                        # 1) If it's hard to tell if an image matches a task from earlier images, then maybe it won't give high values to images until they get close to completing the task
                                        # 2) For the purposes of filtering, even if you can tell from the image what task its doing well before it gets close to completing it, that might be okay, too. Since 
                                        # would still be able to give high values to images that look like they're completing the task, and low values to ones that look like they're doing something else.     
                                            # Maybe has something to do with having a demonstration dataset instead of an RL dataset? 
                            # If it's just the last image, then that would be too sparse? 
                # Without seeing the current image, would it be hard for the Vf to enforce things about temporal consistency, etc.? 
            

            # Vf: current image, goal image, and language instruction?
                # positives: 
                # basically, if the discriminator w/ current image and goal image works, see if you can basically learn that (being able to pick out temporal consistency, etc.) but then also learn 
                # stuff about whether a goal image completes a given task or something. 
                # COuld maybe think of this as learnign the sum of two objectives: discriminate whether an image is generated vs. real or enco

            # Formulations that don't really make sense? 
            # Does just discriminator with one image make any sense? Not really? 
                # No way to know temporal consistency? Other than just learning that there shouldn't be two blue blocks at a time, etc. 
                # Otherwise, could only rely on artifacts, which we don't want 
            
            # Discriminator w/ one image + language instruction? 
                # can potentially use the language instruction to see if what is happening in an image makes sense for accomplishign teh task describved in the language instruction?
                    # basically a language-conditioned contrastive Vf at this point then?
                # Would need to do some ablations to see if it is actually using the language instruction at all or if it is just ignoring it 
            

            # questions: in contrasive RL w a Q fn, how does it not just learn to ignore the actions? 


            # make a script that visualizes discriminator/Vf output along some saved trajectories 


            # observations = jnp.concatenate([batch["observations"], batch["encode_decode"], batch["generated"]], axis=0)
            # goals = jnp.concatenate([batch["goals"], batch["goals"], batch["goals"]], axis=0)

            if self.config["zero_out_obs"]:
                observations = {key:jnp.zeros_like(val) for key, val in batch["observations"].items()}
            else:
                observations = batch["observations"]

            


            rng, key = jax.random.split(rng)
            v = self.state.apply_fn(
                {"params": params},  # gradient flows through here
                (observations, batch["goals"]),
                train=True,
                rngs={"dropout": key},
                name="value",
            )
            

            # batch["uses_generated_goal"]
            # batch["uses_encode_decode_goal"] 
            # batch["uses_noised_encode_decode_goal"]


            labels = jnp.logical_not(batch["uses_generated_goal"])

            bce_loss = optax.sigmoid_binary_cross_entropy(logits=v, labels=labels)
            overall_loss = bce_loss.mean()

            # optax.sigmoid_binary_cross_entropy(logits=v, labels=labels * 1.0).mean()


            generated_goal_mask = batch["uses_generated_goal"].astype(bool)
            encode_decode_mask = batch["uses_encode_decode_goal"].astype(bool)
            noised_encode_decode_mask = batch["uses_noised_encode_decode_goal"].astype(bool)
            real_goal_mask = jnp.logical_not(batch["uses_generated_goal"] + batch["uses_encode_decode_goal"] + batch["uses_noised_encode_decode_goal"])

            def masked_mean(values, mask):
                # return jnp.where(mask, values, 0.0).sum() / mask.sum()
                return (values * mask).sum() / mask.sum()

            real_loss = masked_mean(bce_loss, real_goal_mask)
            generated_loss = masked_mean(bce_loss, generated_goal_mask)
            encode_decode_loss = masked_mean(bce_loss, encode_decode_mask)
            noised_encode_decode_loss = masked_mean(bce_loss, noised_encode_decode_mask)

            overall_logits = v.mean()
            real_logits = masked_mean(v, real_goal_mask)
            generated_logits = masked_mean(v, generated_goal_mask)
            encode_decode_logits = masked_mean(v, encode_decode_mask)
            noised_encode_decode_logits = masked_mean(v, noised_encode_decode_mask)


            overall_pred = jax.nn.sigmoid(v).mean()
            real_pred = masked_mean(jax.nn.sigmoid(v), real_goal_mask)
            generated_pred = masked_mean(jax.nn.sigmoid(v), generated_goal_mask)
            encode_decode_pred = masked_mean(jax.nn.sigmoid(v), encode_decode_mask)
            noised_encode_decode_pred = masked_mean(jax.nn.sigmoid(v), noised_encode_decode_mask)
            

            metrics = {
                "overall_loss": overall_loss,
                "real_loss": real_loss,
                "generated_loss": generated_loss,
                "encode_decode_loss": encode_decode_loss,
                "noised_encode_decode_loss": noised_encode_decode_loss,

                "overall_logits": overall_logits,
                "real_logits": real_logits,
                "generated_logits": generated_logits,
                "encode_decode_logits": encode_decode_logits,
                "noised_encode_decode_logits": noised_encode_decode_logits,

                "overall_pred": overall_pred,
                "real_pred": real_pred,
                "generated_pred": generated_pred,
                "encode_decode_pred": encode_decode_pred,
                "noised_encode_decode_pred": noised_encode_decode_pred,
                
            }


            return overall_loss, metrics


        loss_fns = {
            "value": value_loss_fn,
        }


        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        

        # # update the target params
        # new_state = new_state.target_update(self.config["target_update_rate"])
        info["value_lr"] = self.lr_schedules["value"](self.state.step)

        return self.replace(state=new_state), info
    
    @jax.jit
    def value_function(self, observations, goals):
        v = self.state.apply_fn(
            {"params": self.state.params},
            (observations, goals),
            name="value",
        )

        return v



    @jax.jit
    def get_debug_metrics(self, batch, gripper_close_val=None, **kwargs):

        

        # get the average preds on real, generated, encoded_decoded goals, noised enc, 
        # also the losses on each of these 

        # nn.sigmoid()

        # labels = jnp.logical_not(batch["uses_generated_goal"])

        #     bce_loss = optax.sigmoid_binary_cross_entropy(logits=v, labels=labels)
        #     bce_loss = bce_loss.mean()

        if self.config["zero_out_obs"]:
            observations = {key:jnp.zeros_like(val) for key, val in batch["observations"].items()}
        else:
            observations = batch["observations"]

        v = self.state.apply_fn(
            {"params": self.state.params},
            (observations, batch["goals"]),
            name="value",
        )
        

        labels = jnp.logical_not(batch["uses_generated_goal"])

        bce_loss = optax.sigmoid_binary_cross_entropy(logits=v, labels=labels)

        overall_loss = bce_loss.mean()

        generated_goal_mask = batch["uses_generated_goal"].astype(bool)
        encode_decode_mask = batch["uses_encode_decode_goal"].astype(bool)
        noised_encode_decode_mask = batch["uses_noised_encode_decode_goal"].astype(bool)
        real_goal_mask = jnp.logical_not(batch["uses_generated_goal"] + batch["uses_encode_decode_goal"] + batch["uses_noised_encode_decode_goal"])

        # assert jnp.array_equal(generated_goal_mask + encode_decode_mask + noised_encode_decode_mask + real_goal_mask, jnp.ones_like(generated_goal_mask)).item(), f"generated_goal_mask: {generated_goal_mask}, encode_decode_mask: {encode_decode_mask}, encode_decode_mask: {encode_decode_mask}, noised_encode_decode_mask: {noised_encode_decode_mask}"

        # lambda_operation = lambda x: jnp.where(x > 0, x, 0.0).sum() / (x > 0).sum()
        # real_loss = bce_loss[real_goal_mask].mean()
        # generated_loss = bce_loss[generated_goal_mask].mean()
        # encode_decode_loss = bce_loss[encode_decode_mask].mean()
        # noised_encode_decode_loss = bce_loss[noised_encode_decode_mask].mean()

        # overall_logits = v.mean()
        # real_logits = v[real_goal_mask].mean()
        # generated_logits = v[generated_goal_mask].mean()
        # encode_decode_logits = v[encode_decode_mask].mean()
        # noised_encode_decode_logits = v[noised_encode_decode_mask].mean()


        # overall_pred = jax.nn.sigmoid(v).mean()
        # real_pred = jax.nn.sigmoid(v[real_goal_mask]).mean()
        # generated_pred = jax.nn.sigmoid(v[generated_goal_mask]).mean()
        # encode_decode_pred = jax.nn.sigmoid(v[encode_decode_mask]).mean()
        # noised_encode_decode_pred = jax.nn.sigmoid(v[noised_encode_decode_mask]).mean()

        def masked_mean(values, mask):
            # return jnp.where(mask, values, 0.0).sum() / mask.sum()
            return (values * mask).sum() / mask.sum()



        real_loss = masked_mean(bce_loss, real_goal_mask)
        generated_loss = masked_mean(bce_loss, generated_goal_mask)
        encode_decode_loss = masked_mean(bce_loss, encode_decode_mask)
        noised_encode_decode_loss = masked_mean(bce_loss, noised_encode_decode_mask)

        overall_logits = v.mean()
        real_logits = masked_mean(v, real_goal_mask)
        generated_logits = masked_mean(v, generated_goal_mask)
        encode_decode_logits = masked_mean(v, encode_decode_mask)
        noised_encode_decode_logits = masked_mean(v, noised_encode_decode_mask)

        overall_pred = jax.nn.sigmoid(v).mean()
        real_pred = masked_mean(jax.nn.sigmoid(v), real_goal_mask)
        generated_pred = masked_mean(jax.nn.sigmoid(v), generated_goal_mask)
        encode_decode_pred = masked_mean(jax.nn.sigmoid(v), encode_decode_mask)
        noised_encode_decode_pred = masked_mean(jax.nn.sigmoid(v), noised_encode_decode_mask)
        

        metrics = {
            "overall_loss": overall_loss,
            "real_loss": real_loss,
            "generated_loss": generated_loss,
            "encode_decode_loss": encode_decode_loss,
            "noised_encode_decode_loss": noised_encode_decode_loss,

            "overall_logits": overall_logits,
            "real_logits": real_logits,
            "generated_logits": generated_logits,
            "encode_decode_logits": encode_decode_logits,
            "noised_encode_decode_logits": noised_encode_decode_logits,

            "overall_pred": overall_pred,
            "real_pred": real_pred,
            "generated_pred": generated_pred,
            "encode_decode_pred": encode_decode_pred,
            "noised_encode_decode_pred": noised_encode_decode_pred,
            
        }

        # for key, val in metrics.items():
        #     print(f"{key}: {val}")

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        goals: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        shared_goal_encoder: bool = True,
        early_goal_concat: bool = False,
        use_proprio: bool = False,
        # negative_proportion: float = 0.0,
        network_kwargs: dict = {"hidden_dims": [256, 256], "dropout": 0.0},
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        # Algorithm config
        # discount=0.95,
        # expectile=0.9,
        # temperature=1.0,
        # target_update_rate=0.002,
        # dropout_target_networks=True,
        zero_out_obs: bool = False,
    ):
        if early_goal_concat:
            # passing None as the goal encoder causes early goal concat
            goal_encoder_def = None
        else:
            if shared_goal_encoder:
                goal_encoder_def = encoder_def
            else:
                goal_encoder_def = copy.deepcopy(encoder_def)


        encoder_def = GCEncodingWrapper(
            encoder=encoder_def,
            goal_encoder=goal_encoder_def,
            use_proprio=use_proprio,
            stop_gradient=False,
        )

        if shared_encoder:
            encoders = {
                "value": encoder_def,
            }
        else:
            # I (kvablack) don't think these deepcopies will break
            # shared_goal_encoder, but I haven't tested it.
            encoders = {
                "value": encoder_def,
            }

        # network_kwargs["activate_final"] = True
        network_kwargs["activate_final"] = False ### ??? ###
        
        networks = {
            "value": ValueCritic(encoders["value"], MLP(**network_kwargs)),
        }

        model_def = ModuleDict(networks)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            value=[(observations, goals)],
        )["params"]

        # no decay
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {
            "value": lr_schedule,
        }
        
        txs = {k: optax.adam(v) for k, v in lr_schedules.items()}

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                # discount=discount,
                # temperature=temperature,
                # target_update_rate=target_update_rate,
                # expectile=expectile,
                # dropout_target_networks=dropout_target_networks,
                # negative_proportion=negative_proportion,
                zero_out_obs=zero_out_obs,
            )
        )
        return cls(state, config, lr_schedules)
