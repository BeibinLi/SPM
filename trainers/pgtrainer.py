import pdb
import random
import torch

from model_utils import calc_probs_log_probs
from trainers.utils import PolicyTrainer, compute_advantage, compile_from_log, update_critic

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
            advantages = keyword_dict["advantage"].to(self.model.device)

            _true_log_probs = keyword_dict["log_prob"]

            # PG uses log probs
            probs_log_probs = calc_probs_log_probs(
                model=self.model,
                tokens=input_tokens,
                attention_mask=attention_mask,
                generated_mask=generated_mask,
                generation_config=self.generation_config,
                calc_probs=False,
                calc_log_probs=True)
            log_probs = probs_log_probs["log_probs"]

            if torch.norm(_true_log_probs - log_probs.data.cpu()) > 1:
                print(_true_log_probs, log_probs.data)
                pdb.set_trace()

            # print(_true_log_probs, log_probs.data, advantages)

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

        # print(losses)

        # Equal to average per step loss
        return {"loss": sum(losses), "critic_loss": critic_loss}
