import pdb
import random
import torch

from statistics import mean

from model_utils import calc_probs_log_probs

MAX_VAL = 100

def compile_from_log(generation_result_steps, pad_token_id):
    STACK_KEYS = ["tokens", "attention_mask"]
    TENSOR_KEYS = [("Q_value", torch.float32), ("generated_mask", torch.bool),
                   ("advantage", torch.float32), ("prob", torch.float32)]
    LIST_KEYS = ["prompt", "scores"]
    KEYS = STACK_KEYS + [k[0] for k in TENSOR_KEYS] + LIST_KEYS

    ret = {k: [] for k in KEYS}
    for step in generation_result_steps:
        for key in KEYS:
            if key in step:
                ret[key].append(step[key])

    # Must contain 'tokens'
    device = ret["tokens"][0].device
    max_len = max([x.shape[0] for x in ret["tokens"]])

    # Pad to same length
    for i in range(len(ret["tokens"])):
        cur_len = ret["tokens"][i].shape[0]
        ret["tokens"][i] = torch.cat(
            (torch.full((max_len - cur_len,),
                        pad_token_id,
                        device=ret["tokens"][i].device), ret["tokens"][i]))
        ret["attention_mask"][i] = torch.cat((torch.full(
            (max_len - cur_len,), 0,
            device=ret["attention_mask"][i].device), ret["attention_mask"][i]))
        ret["generated_mask"][i] = [False] * (
            max_len - cur_len) + ret["generated_mask"][i]

    for key in STACK_KEYS:
        if key in ret:
            ret[key] = torch.stack(ret[key])

    for key, dtype in TENSOR_KEYS:
        if key in ret:
            ret[key] = torch.tensor(ret[key], dtype=dtype, device=device)

    return ret


def compute_advantage(data, batch_size, critic_model, tokenizer):
    for i in range(0, len(data), batch_size):
        # Get the input batch for this step
        keyword_dict = compile_from_log(data[i:i + batch_size],
                                        tokenizer.pad_token_id)
        prompts = keyword_dict["prompt"]

        if critic_model is not None:
            value_inputs = tokenizer.batch_encode_plus(
                prompts,
                truncation=True,
                padding=True,
                max_length=critic_model.config.max_length,
                return_tensors="pt")
            value_inputs = {
                k: v.to(critic_model.device) for k, v in value_inputs.items()
            }

            with torch.no_grad():
                values = critic_model(**value_inputs).logits.squeeze(-1)
        else:
            values = torch.zeros(len(prompts))

        j = 0
        for d in data[i:i + batch_size]:
            d["advantage"] = d["Q_value"] - values[j].item()
            j += 1


def update_critic(data, critic_model, critic_optimizer, tokenizer, batch_size,
                  max_grad_norm, gradient_accumulation_steps, update_iter):
    losses = []

    accumulated_steps = 0
    critic_optimizer.zero_grad()

    for iter in range(update_iter):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            # Get the input batch for this step
            keyword_dict = compile_from_log(data[i:i + batch_size],
                                            tokenizer.pad_token_id)
            Q_values = keyword_dict["Q_value"]
            prompts = keyword_dict["prompt"]

            value_inputs = tokenizer.batch_encode_plus(
                prompts,
                truncation=True,
                padding=True,
                max_length=critic_model.config.max_length,
                return_tensors="pt")
            value_inputs = {
                k: v.to(critic_model.device) for k, v in value_inputs.items()
            }
            values = critic_model(**value_inputs).logits.squeeze(-1)

            critic_optimizer.zero_grad()

            loss = torch.nn.MSELoss()(Q_values.to(critic_model.device) / MAX_VAL,
                                      values / MAX_VAL) / gradient_accumulation_steps
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                pdb.set_trace()
            loss.backward()
            losses.append(loss.item())
            accumulated_steps += 1

            if accumulated_steps == gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(critic_model.score.parameters(),
                                               max_grad_norm)
                critic_optimizer.step()
                accumulated_steps = 0
                critic_optimizer.zero_grad()

    if accumulated_steps:
        torch.nn.utils.clip_grad_norm_(critic_model.score.parameters(), max_grad_norm)
        critic_optimizer.step()

    return mean(losses)


class PolicyTrainer:

    def __init__(self, model, tokenizer, optimizer, gradient_accumulation_steps,
                 generation_config, critic_model, critic_optimizer,
                 critic_update_freq, critic_update_iter, batch_size,
                 max_grad_norm, **kwargs):
        """
        Compute gradient for proximal policy optimization.

        Args:
        - `model` (PeftModel): the model to be updated
        - `tokenizer` (AutoTokenizer): the tokenizer for `model`
        - `optimizer` (torch.optim.Optimizer): the optimizer for `model`
        - `generation_config` (GenerationConfig): the generation config used to
        generate the dialog
        - `generation_results` (list): the generation result, which is a list
        consisting of `batch_size` lists. Each inner list contains dicts in the
        following format:
        {
            "tokens": torch.Tensor,
            "generated_mask": list,
            "attention_mask": torch.Tensor,
            "cost": float,
            "Q_value": float,
            "step": int
        }
        - `critic_model` (PeftModel): the value model to be updated
        - `critic_tokenizer` (AutoTokenizer): the tokenizer for `critic_model`
        - `critic_optimizer` (torch.optim.Optimizer): the optimizer for
        `critic_model`
        - `clip_coef` (float): the clipping coefficient for PPO
        - `max_grad_norm` (float): the maximum gradient norm for gradient clipping
        """

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.generation_config = generation_config
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
        self.use_critic = critic_model is not None

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.data = []
        self.gradient_accumulated_steps = 0

        self.critic_update_freq = critic_update_freq
        self.critic_update_iter = critic_update_iter
        self.critic_data = []
        self.critic_accumulated_steps = 0


class PGTrainer(PolicyTrainer):

    def train(self, generation_results):
        if generation_results == []:
            return {"loss": None, "critic_loss": None}

        self.data += generation_results
        self.gradient_accumulated_steps += 1
        if self.gradient_accumulated_steps < self.gradient_accumulation_steps:
            return {"loss": None, "critic_loss": None}

        self.critic_data += generation_results
        self.critic_accumulated_steps += 1

        self.optimizer.zero_grad()

        losses = []
        data = [x for r in self.data for x in r]

        compute_advantage(data=data,
                          batch_size=self.batch_size,
                          critic_model=self.critic_model,
                          tokenizer=self.tokenizer)

        random.shuffle(data)
        for i in range(0, len(data), self.batch_size):
            # Get the input batch for this step
            keyword_dict = compile_from_log(
                generation_result_steps=data[i:i + self.batch_size],
                pad_token_id=self.tokenizer.pad_token_id)
            input_tokens = keyword_dict["tokens"]
            attention_mask = keyword_dict["attention_mask"]
            generated_mask = keyword_dict["generated_mask"]
            advantages = keyword_dict["advantage"]

            # PG uses log probs
            with torch.autocast(device_type="cuda"):
                probs_log_probs = calc_probs_log_probs(
                    model=self.model,
                    tokens=input_tokens,
                    attention_mask=attention_mask,
                    generated_mask=generated_mask,
                    generation_config=self.generation_config,
                    calc_probs=True,
                    calc_log_probs=False)
                log_probs = probs_log_probs["probs"]

                loss = torch.sum(advantages * log_probs) / len(data)
            loss.backward()
            losses.append(loss.item())

        # Policy network update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.max_grad_norm)
            
        self.optimizer.step()

        self.gradient_accumulated_steps = 0
        self.data = []

        critic_loss = None
        if (self.use_critic
                and self.critic_accumulated_steps == self.critic_update_freq):
            critic_loss = update_critic(
                data=[x for r in self.critic_data for x in r],
                critic_model=self.critic_model,
                critic_optimizer=self.critic_optimizer,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                max_grad_norm=self.max_grad_norm,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                update_iter=self.critic_update_iter)

            self.critic_accumulated_steps = 0
            self.critic_data = []

        # Equal to average per step loss
        return {"loss": sum(losses), "critic_loss": critic_loss}


class PPOTrainer(PolicyTrainer):

    def __init__(self, ppo_clip_coef, ppo_update_iter, **kwargs):
        super().__init__(**kwargs)

        assert self.use_critic, "For PPO, critic model must be provided."

        self.ppo_clip_coef = ppo_clip_coef
        self.ppo_update_iter = ppo_update_iter

    def train(self, generation_results):
        if generation_results == []:
            return {"loss": None, "critic_loss": None}

        self.data += generation_results
        self.gradient_accumulated_steps += 1
        if self.gradient_accumulated_steps < self.gradient_accumulation_steps:
            return {"loss": None, "critic_loss": None}

        self.critic_data += generation_results
        self.critic_accumulated_steps += 1

        losses = []
        data = [x for r in self.data for x in r]

        compute_advantage(data=data,
                          batch_size=self.batch_size,
                          critic_model=self.critic_model,
                          tokenizer=self.tokenizer)

        for iter in range(self.ppo_update_iter):
            self.optimizer.zero_grad()

            random.shuffle(data)
            for i in range(0, len(data), self.batch_size):
                # Get the input batch for this step
                keyword_dict = compile_from_log(
                    generation_result_steps=data[i:i + self.batch_size],
                    pad_token_id=self.tokenizer.pad_token_id)
                input_tokens = keyword_dict["tokens"]
                attention_mask = keyword_dict["attention_mask"]
                generated_mask = keyword_dict["generated_mask"]
                advantages = keyword_dict["advantage"]
                old_probs = keyword_dict["prob"]

                # PPO uses probs
                probs_log_probs = calc_probs_log_probs(
                    model=self.model,
                    tokens=input_tokens,
                    attention_mask=attention_mask,
                    generated_mask=generated_mask,
                    generation_config=self.generation_config,
                    calc_probs=True,
                    calc_log_probs=False)
                probs = probs_log_probs["probs"]

                # Advantage for minimizing cost is negative of maximizing reward
                loss1 = probs / old_probs * (-advantages)
                loss2 = torch.clamp(probs / old_probs, 1 - self.ppo_clip_coef,
                                    1 + self.ppo_clip_coef) * (-advantages)
                loss = torch.sum(-torch.min(loss1, loss2)) / len(data)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    pdb.set_trace()
                losses.append(loss.item())

                loss.backward()

            # Policy network update
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.max_grad_norm)
            self.optimizer.step()

        self.gradient_accumulated_steps = 0
        self.data = []

        critic_loss = None
        if (self.use_critic
                and self.critic_accumulated_steps == self.critic_update_freq):
            critic_loss = update_critic(
                data=[x for r in self.critic_data for x in r],
                critic_model=self.critic_model,
                critic_optimizer=self.critic_optimizer,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                max_grad_norm=self.max_grad_norm,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                update_iter=self.critic_update_iter)

            self.critic_accumulated_steps = 0
            self.critic_data = []

        # Equal to average per step loss
        return {
            "loss": sum(losses) / self.ppo_update_iter,
            "critic_loss": critic_loss
        }


TRAINERS = {
    "pg": PGTrainer,
    "ppo": PPOTrainer,
}
