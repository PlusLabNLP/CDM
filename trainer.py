import datasets  # This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from models.modeling_t5cdm import T5ForContrastiveDistributionModeling
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset
import torch.utils.data as datautils
import torch
import numpy as np
import argparse
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer import is_sagemaker_mp_enabled, version, is_torch_tpu_available, Dict
import tqdm

def compute_metrics(eval_pred):
    losses = eval_pred.predictions
    return {
        "amateur_loss": losses[:, 0, 0].mean(axis=0),
        "expert_loss": losses[:, 1, 0].mean(axis=0),
        "discrepancy": losses[:, 0, 0].mean(axis=0) - losses[:, 1, 0].mean(axis=0)
    }
class CustomTrainer(Trainer):
    # def training_step(self, model: torch.nn.Module, inputs):
    #     """
    #     Perform a training step on a batch of inputs.
    #
    #     Subclass and override to inject custom behavior.
    #
    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #
    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs, is_train=True)
    #     # print(type(self.compute_loss_context_manager()))
    #
    #     return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        input_ids, attention_mask, labels_, decoder_attention_mask = inputs.get("input_ids"), inputs.get(
            "attention_mask"), inputs.get("labels"), inputs.get("decoder_attention_mask")
        encoder_max_len = attention_mask.sum(dim=-1).amax(dim=-1)
        decoder_max_len = decoder_attention_mask.sum(dim=-1).amax(dim=-1)
        labels = labels_[:, 0:decoder_max_len].contiguous()
        with torch.no_grad():
            model_output = model(
                input_ids=input_ids[:, 0:encoder_max_len].contiguous(),
                attention_mask=attention_mask[:, 0:encoder_max_len].contiguous(),
                labels=labels,
            )
            decoder_attention_mask = decoder_attention_mask[:, 0:decoder_max_len].contiguous().to(
                model_output.logits.dtype)
            amateur_loss = -model_output.likelihoods[0].log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(
                dim=-1)).reshape_as(labels) * decoder_attention_mask
            expert_loss = -model_output.likelihoods[1].log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(
                dim=-1)).reshape_as(labels) * decoder_attention_mask
            logits = torch.stack((amateur_loss.sum(dim=-1), expert_loss.sum(dim=-1))).unsqueeze(dim=0)
            # loss = amateur_loss.sum(dim=-1).mean(dim=0) + expert_loss.sum(dim=-1).mean(dim=0)

        return (None, logits, labels)
    def compute_loss(self, model, inputs, return_outputs=False, is_train=False):

        input_ids, attention_mask, labels, decoder_attention_mask = inputs.get("input_ids"), inputs.get("attention_mask"), inputs.get("labels"), inputs.get("decoder_attention_mask")
        encoder_max_len = attention_mask.sum(dim=-1).amax(dim=-1)
        decoder_max_len = decoder_attention_mask.sum(dim=-1).amax(dim=-1)
        labels = labels[:, 0:decoder_max_len].contiguous()
        model_output = model(
            input_ids=input_ids[:, 0:encoder_max_len].contiguous(),
            attention_mask=attention_mask[:, 0:encoder_max_len].contiguous(),
            labels=labels,
        )

        decoder_attention_mask = decoder_attention_mask[:, 0:decoder_max_len].contiguous().to(
            model_output.logits.dtype)
        amateur_loss = -model_output.likelihoods[0].log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(
            dim=-1)).reshape_as(labels) * decoder_attention_mask
        expert_loss = -model_output.likelihoods[1].log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(
            dim=-1)).reshape_as(labels) * decoder_attention_mask

        composed_loss = amateur_loss.sum(dim=-1) + expert_loss.sum(dim=-1)

        with torch.no_grad():
            if hasattr(self.state, 'most_recent_output') and\
                "amateur_loss" in self.state.most_recent_output and\
                "expert_loss" in self.state.most_recent_output and\
                "n" in self.state.most_recent_output:
                n = self.state.most_recent_output["n"]
                self.state.most_recent_output = {
                    "amateur_loss": (1 / (n + 1.)) * amateur_loss.sum(dim=-1).mean(dim=0).cpu().item() + (n / (n + 1.)) * self.state.most_recent_output["amateur_loss"],
                    "expert_loss":  (1 / (n + 1.)) * expert_loss.sum(dim=-1).mean(dim=0).cpu().item() + (n / (n + 1.)) * self.state.most_recent_output["expert_loss"],
                    "n": n + 1,
                }
            else:
                self.state.most_recent_output = {
                    "amateur_loss": amateur_loss.sum(dim=-1).mean(dim=0).cpu().item(),
                    "expert_loss": expert_loss.sum(dim=-1).mean(dim=0).cpu().item(),
                    "n": 1
                }

        return (composed_loss.mean(), model_output) if return_outputs else composed_loss.mean()
class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs:dict=None, **kwargs):
        if state.is_local_process_zero:
            try:
                _logs = {
                        "epoch": logs["epoch"],
                        "total_loss": logs["loss"],
                        "LR": logs["learning_rate"],
                }
            except KeyError:
                return
            logs.clear()
            try:
                logs.update({
                    "epoch": _logs["epoch"],
                    "step": state.global_step,
                    "LR": _logs["LR"],
                    "amateur_loss": state.most_recent_output["amateur_loss"],
                    "expert_loss": state.most_recent_output["expert_loss"]
                })
                state.most_recent_output = dict()
            except ValueError:
                print(state.most_recent_output)

class DialogueFIIMDataset(Dataset):
    def __init__(self, tokenizer: T5Tokenizer, max_len=1024):
        self.dialogues = None
        self.tokenizer = tokenizer
        self.max_len = max_len


    def add_from_file(self, file_path):
        if self.dialogues is None:
            self.dialogues = np.array([line.strip() for line in open(file_path, "r").readlines()])
        else:
            self.dialogues = np.concatenate((self.dialogues, np.array([line.strip() for line in open(file_path, "r").readlines()])), axis=0)

    def __len__(self):
        return self.dialogues.shape[0]

    def size(self):
        return self.__len__()

    def __getitem__(self, item):
        dialogue = self.dialogues[item]
        utts = dialogue.split("</UTT>")
        total_len = len(utts)
        idx_recons = np.random.randint(0, total_len)
        while len(self.tokenizer.tokenize(utts[idx_recons])) == 0:
            idx_recons = np.random.randint(0, total_len)
        if np.random.randint(0, 3) == 0:
            context = utts[0:idx_recons] + ["<extra_id_0>"] + utts[idx_recons+1:]
            recons = "<extra_id_0>" + utts[idx_recons]
        else:
            inner_utt = utts[idx_recons]
            inner_utt_tokens = self.tokenizer.tokenize(inner_utt)
            inner_len = len(inner_utt_tokens)
            inner_l, inner_r = sorted([np.random.randint(0, inner_len), np.random.randint(0, inner_len)])
            inner_context = self.tokenizer.convert_tokens_to_string(inner_utt_tokens[0:inner_l] + ["<extra_id_0>"] + inner_utt_tokens[inner_r:])
            context = utts[0:idx_recons] + [inner_context] + utts[idx_recons+1:]
            recons = self.tokenizer.convert_tokens_to_string(["<extra_id_0>"] + inner_utt_tokens[inner_l+1:inner_r])

        encoder_input_ids = self.tokenizer("</UTT>".join(context), return_tensors="pt", max_length=self.max_len, padding="max_length")
        decoder_labels = self.tokenizer(recons, return_tensors="pt", max_length=self.max_len, padding="max_length")

        return {
            "input_ids": encoder_input_ids.input_ids[0],
            "attention_mask": encoder_input_ids.attention_mask[0],
            "labels": decoder_labels.input_ids[0],
            "decoder_attention_mask": decoder_labels.attention_mask[0]
        }

class CommonGenDataset(Dataset):
    def __init__(self, tokenizer, dataset, split='train', max_len=256):
        self.tokenizer = tokenizer
        self.huggingface_instance = dataset
        self.split = split
        self.max_len = max_len

    def __len__(self):
        return len(self.huggingface_instance[self.split])

    def size(self):
        return self.__len__()

    def __getitem__(self, item):
        line = self.huggingface_instance[self.split][item]
        concepts = ", ".join(line['concepts'])
        target = line['target']

        encoder_input_ids = self.tokenizer(concepts, return_tensors="pt", max_length=self.max_len,
                                           padding="max_length")
        decoder_labels = self.tokenizer(target, return_tensors="pt", max_length=self.max_len, padding="max_length")

        return {
            "input_ids": encoder_input_ids.input_ids[0],
            "attention_mask": encoder_input_ids.attention_mask[0],
            "labels": decoder_labels.input_ids[0],
            "decoder_attention_mask": decoder_labels.attention_mask[0]
        }
class CommonGenFIIMDataset(Dataset):
    def __init__(self, tokenizer, dataset, split='train', max_len=256):
        self.tokenizer = tokenizer
        self.huggingface_instance = dataset
        self.split = split
        self.max_len = max_len

    def __len__(self):
        return len(self.huggingface_instance[self.split])

    def size(self):
        return self.__len__()

    def __getitem__(self, item):
        line = self.huggingface_instance[self.split][item]
        concepts = " ".join(line['concepts'])
        target = line['target']
        if len(self.tokenizer.tokenize(target)) < 4:
            context = concepts + " = <extra_id_0>"
            target = self.tokenizer.convert_tokens_to_string(['<extra_id_0>'] + self.tokenizer.tokenize(target))
        else:
            tokenized = self.tokenizer.tokenize(target)
            i, j = sorted([np.random.randint(0, len(tokenized)),np.random.randint(0,len(tokenized))])
            context = concepts + " = " + self.tokenizer.convert_tokens_to_string(tokenized[0:i-1] + ['<extra_id_0>'] + tokenized[j+1:])
            target = self.tokenizer.convert_tokens_to_string(['<extra_id_0>'] + tokenized[i:j])

        encoder_input_ids = self.tokenizer(context, return_tensors="pt", max_length=self.max_len,
                                           padding="max_length")
        decoder_labels = self.tokenizer(target, return_tensors="pt", max_length=self.max_len, padding="max_length")

        return {
            "input_ids": encoder_input_ids.input_ids[0],
            "attention_mask": encoder_input_ids.attention_mask[0],
            "labels": decoder_labels.input_ids[0],
            "decoder_attention_mask": decoder_labels.attention_mask[0]
        }
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--amateur",
        default="small",
        type=str,
        required=False,
        help="Amateur model size"
    )
    parser.add_argument(
        "--expert",
        default="large",
        type=str,
        required=False,
        help="Expert model size"
    )
    parser.add_argument(
        "--dataset",
        default="dialogue",
        type=str,
        required=False,
        help="dataset choice"
    )
    parser.add_argument(
        "--legacy_training",
        default=False,
        type=str2bool,
        required=False,
        help="use single GPU training"
    )
    parser.add_argument(
        "--logsum",
        default=False,
        type=str2bool,
        required=False,
        help="use single GPU training"
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        required=False,
        help="batch size for training"
    )
    parser.add_argument(
        "--iter_per",
        default=256,
        type=int,
        required=False,
        help="gradient accumulation steps"
    )
    parser.add_argument(
        "--local-rank",
        default=-1,
        type=int,
        required=False,
        help="node id",
    )


    args = parser.parse_args()
    model = T5ForContrastiveDistributionModeling(config=[
        T5Config.from_pretrained("t5-%s" % args.amateur), T5Config.from_pretrained("t5-%s" % args.expert),
    ])
    model.models[0].load_state_dict(T5ForConditionalGeneration.from_pretrained("t5-%s" % args.amateur).state_dict())
    model.models[1].load_state_dict(T5ForConditionalGeneration.from_pretrained("t5-%s" % args.expert).state_dict())

    tokenizer = T5Tokenizer.from_pretrained("t5-%s" % args.amateur, legacy=False)

    if args.dataset == "dialogue":
        dataset = DialogueFIIMDataset(tokenizer, max_len=4096)
        dataset.add_from_file("./data/dialogue/clean_train.txt")
        valid_dataset = DialogueFIIMDataset(tokenizer, max_len=4096)
        valid_dataset.add_from_file("./data/dialogue/clean_valid.txt")
    elif args.dataset == "common_gen":
        full_dataset = datasets.load_dataset("common_gen")
        dataset = CommonGenFIIMDataset(tokenizer, dataset=full_dataset, split="train")
        valid_dataset = CommonGenFIIMDataset(tokenizer, dataset=full_dataset, split="validation")
    elif args.dataset == "common_gen_full":
        full_dataset = datasets.load_dataset("common_gen")
        dataset = CommonGenDataset(tokenizer, dataset=full_dataset, split="train")
        valid_dataset = CommonGenDataset(tokenizer, dataset=full_dataset, split="validation")


    if args.legacy_training:
        dataloader = datautils.DataLoader(
            dataset, batch_size=args.batch_size // args.iter_per, shuffle=True, drop_last=False, pin_memory=True,
            num_workers=8
        )
        valid_dataloader = datautils.DataLoader(
            valid_dataset, batch_size=args.batch_size // args.iter_per, shuffle=True, drop_last=False, pin_memory=True,
            num_workers=8
        )
        opt = torch.optim.AdamW(lr=1e-5, params=model.parameters())
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        for model_ in model.models:
            model_.parallelize()

        if args.logsum:
            model.models[-1].requires_grad_(False)
        # model.gradient_checkpointing_enable()
        for epoch_idx in range(50):
            iterator = tqdm.tqdm(dataloader)
            AMATEUR_NLL_REDUCED = []
            EXPERT_NLL_REDUCED = []

            for iter_idx, instance in enumerate(iterator):
                input_ids = instance["input_ids"]
                attention_mask = instance["attention_mask"]
                encoder_max_len = attention_mask.sum(dim=-1).amax(dim=-1)
                labels = instance["labels"]
                decoder_attention_mask = instance["decoder_attention_mask"]
                decoder_max_len = decoder_attention_mask.sum(dim=-1).amax(dim=-1)
                labels = labels[:,0:decoder_max_len].cuda().contiguous()
                model_output = model(
                    input_ids=input_ids[:,0:encoder_max_len].cuda().contiguous(),
                    attention_mask=attention_mask[:,0:encoder_max_len].cuda().contiguous(),
                    labels=labels,
                )
                if iter_idx % args.iter_per == 0:
                    opt.zero_grad()

                decoder_attention_mask = decoder_attention_mask[:,0:decoder_max_len].cuda().contiguous().to(model_output.logits.dtype)

                amateur_loss = -model_output.likelihoods[0].log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(dim=-1)).reshape_as(labels) * decoder_attention_mask
                expert_loss = -model_output.likelihoods[1].log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(dim=-1)).reshape_as(labels) * decoder_attention_mask

                if args.logsum:
                    composed_loss = -(model_output.likelihoods[0] + model_output.likelihoods[0]).log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(dim=-1)).reshape_as(labels) * decoder_attention_mask
                    expert_loss = composed_loss
                    amateur_loss = composed_loss
                else:
                    composed_loss = amateur_loss.sum(dim=-1) + expert_loss.sum(dim=-1)

                (composed_loss.mean() / args.iter_per).backward()
                AMATEUR_NLL_REDUCED.append(amateur_loss.sum(dim=-1).mean().cpu().detach().item())
                EXPERT_NLL_REDUCED.append(expert_loss.sum(dim=-1).mean().cpu().detach().item())

                if (iter_idx + 1) % args.iter_per == 0:
                    opt.step()
                    if (iter_idx // args.iter_per) % 10 == 0:
                        iterator.write("Iteration %d-%d: a-loss %f, e-loss %f" % (epoch_idx, iter_idx // args.iter_per, np.mean(AMATEUR_NLL_REDUCED), np.mean(EXPERT_NLL_REDUCED)))

            iterator = tqdm.tqdm(valid_dataloader)
            AMATEUR_NLL_REDUCED = []
            EXPERT_NLL_REDUCED = []

            for iter_idx, instance in enumerate(iterator):
                input_ids = instance["input_ids"]
                attention_mask = instance["attention_mask"]
                encoder_max_len = attention_mask.sum(dim=-1).amax(dim=-1)
                labels = instance["labels"]
                decoder_attention_mask = instance["decoder_attention_mask"]
                decoder_max_len = decoder_attention_mask.sum(dim=-1).amax(dim=-1)
                labels = labels[:, 0:decoder_max_len].cuda().contiguous()
                with torch.no_grad():
                    model_output = model(
                        input_ids=input_ids[:, 0:encoder_max_len].cuda().contiguous(),
                        attention_mask=attention_mask[:, 0:encoder_max_len].cuda().contiguous(),
                        labels=labels,
                    )
                    decoder_attention_mask = decoder_attention_mask[:, 0:decoder_max_len].cuda().contiguous().to(
                        model_output.logits.dtype)
                    amateur_loss = -model_output.likelihoods[0].log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(
                        dim=-1)).reshape_as(labels) * decoder_attention_mask
                    expert_loss = -model_output.likelihoods[1].log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(
                        dim=-1)).reshape_as(labels) * decoder_attention_mask
                    if args.logsum:
                        composed_loss = -(model_output.likelihoods[0] + model_output.likelihoods[0]).log_softmax(
                            dim=-1).gather(dim=-1, index=labels.unsqueeze(dim=-1)).reshape_as(
                            labels) * decoder_attention_mask
                        expert_loss = composed_loss
                        amateur_loss = composed_loss

                    AMATEUR_NLL_REDUCED.append(amateur_loss.sum(dim=-1).mean().cpu().detach().item())
                    EXPERT_NLL_REDUCED.append(expert_loss.sum(dim=-1).mean().cpu().detach().item())

            iterator.write("Epoch Validation %d: a-loss %f, e-loss %f" % (
            epoch_idx, np.mean(AMATEUR_NLL_REDUCED), np.mean(EXPERT_NLL_REDUCED)))

            if args.dataset == "dialogue":
                model.save_pretrained("checkpoints/CDM%s-%s/checkpoint-%d" % (args.amateur, args.expert, epoch_idx))

            else:
                model.save_pretrained("checkpoints/%s/CDM%s-%s/checkpoint-%d" % (args.dataset, args.amateur, args.expert, epoch_idx))
    else:
        training_args = TrainingArguments(
            output_dir="checkpoints/CDM%s-%s" % (args.amateur, args.expert) if args.dataset == "dialogue" else "checkpoints/%s/CDM%s-%s" % (args.dataset, args.amateur, args.expert),
            num_train_epochs=8,
            label_names=["input_ids", "attention_mask", "labels", "decoder_attention_mask"],
            per_device_train_batch_size=args.batch_size // torch.cuda.device_count() // args.iter_per,
            per_device_eval_batch_size=args.batch_size // torch.cuda.device_count() // args.iter_per,
            warmup_steps=10,
            weight_decay=0.02,
            learning_rate=1e-5,
            adam_beta1=0.9,
            adam_epsilon=1e-6,
            adam_beta2=0.98,
            load_best_model_at_end=True,
            metric_for_best_model='discrepancy',  # use the evaluation loss to save best models
            logging_dir='checkpoints/' + "log-CDM-%s-%s.txt" % (args.amateur, args.expert),
            logging_steps=10,
            tf32=True,
            # fsdp="full_shard",
            gradient_accumulation_steps=args.iter_per,
            dataloader_num_workers=24,
            ddp_find_unused_parameters=False,
            # half_precision_backend="cuda_amp",
            evaluation_strategy="epoch",
            local_rank=args.local_rank if args.local_rank is not None else -1,
            # fsdp="full_shard" if args.local_rank is not None else False,
            optim="adamw_torch",
            # gradient_checkpointing=True,
            save_strategy="epoch",
            save_total_limit=100)
        # training_args.rescued_log = None
        # print(training_args)
        open(training_args.logging_dir, "w").close()
        trainer = CustomTrainer(model=model, args=training_args, train_dataset=dataset, eval_dataset=valid_dataset,
                                tokenizer=tokenizer, compute_metrics=compute_metrics, callbacks=[LogCallback()])

        # trainer.args.trainer = trainer
        # print(training_args)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.local_rank is None or args.local_rank == 0:
            print("New training started. Creating log file.")
            print("Total Trainable Parameters: ", pytorch_total_params)
            open(trainer.args.output_dir + "/log", "w").close()

        trainer.train()

if __name__ == '__main__':
    main()
