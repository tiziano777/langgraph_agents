import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import json
from json_repair import repair_json
from datetime import datetime
import os

class GenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset_for_inference,
                 info=None,
                 num_examples=10, log_steps_interval=20, max_new_tokens=200):

        self.model = model
        self.tokenizer = tokenizer
        self.training_args_info = info  # Oggetto di tipo TrainingArguments

        # Crea il file di log con timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, f"training_log_{timestamp_str}.txt")

        # Logging function
        def _log_to_file(msg):
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")
            print(msg)

        self._log = _log_to_file

        # ✍️ Scrivi i TrainingArguments all'inizio del file di log
        self._log("=" * 80)
        self._log("TRAINING ARGUMENTS:\n")
        if self.training_args_info is not None:
            self._log(self.training_args_info)

        else:
            self._log("No training arguments provided.")
        self._log("=" * 80 + "\n")

        # Prepara i prompt
        selected_examples = eval_dataset_for_inference.select(range(min(num_examples, len(eval_dataset_for_inference))))
        self.prompts_for_generation = []
        self.ground_truths = []
        end_instruction_token_str = "[/INST]"

        for example in selected_examples:
            full_sequence = example["text"]
            inst_end_pos = full_sequence.rfind(end_instruction_token_str)

            if inst_end_pos != -1:
                inference_input = full_sequence[:inst_end_pos + len(end_instruction_token_str)].strip()
                inference_input += self.tokenizer.bos_token
                assistant_response_start_pos = inst_end_pos + len(end_instruction_token_str)
                prompt_text_for_gen = full_sequence[:assistant_response_start_pos].strip()
                gt_text = full_sequence[assistant_response_start_pos:].strip()
                if gt_text.endswith(self.tokenizer.eos_token):
                    gt_text = gt_text[:-len(self.tokenizer.eos_token)].strip()
                self.prompts_for_generation.append(prompt_text_for_gen)
                self.ground_truths.append(gt_text)
            else:
                self._log(f"Warning: Could not find '{end_instruction_token_str}' in example: {full_sequence[:100]}...")
                self.prompts_for_generation.append("")
                self.ground_truths.append("")

        self.num_examples = len(self.prompts_for_generation)
        self.log_steps_interval = log_steps_interval
        self.printed_steps = set()
        self.max_new_tokens = max_new_tokens

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _generate_output(self, prompt_text):
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            max_length=self.tokenizer.model_max_length
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            temperature=0.1,
            top_p=0.5,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        generated_output_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_output_ids, skip_special_tokens=True).strip()

        output_prefix = "Output:\n"
        if generated_text.startswith(output_prefix):
            generated_text = generated_text[len(output_prefix):].strip()

        if generated_text.endswith(self.tokenizer.eos_token):
            generated_text = generated_text[:-len(self.tokenizer.eos_token)].strip()

        return generated_text

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_steps_interval == 0 and state.global_step not in self.printed_steps:
            self._log(f"\n--- Generation Examples at Step {state.global_step} ---")
            self.model.eval()
            with torch.no_grad():
                for i in range(self.num_examples):
                    prompt = self.prompts_for_generation[i]
                    ground_truth = self.ground_truths[i]
                    generated_output_raw = self._generate_output(prompt)

                    self._log(f"\n--- Example {i+1} ---")
                    self._log(f"Prompt (inference input):\n{prompt[-2500:]}")
                    self._log(f"Ground Truth Output:\n{ground_truth}")
                    self._log(f"Generated Output (raw):\n{generated_output_raw}")

                    try:
                        gt_parsed_json = json.loads(ground_truth)
                        gen_parsed_json = repair_json(generated_output_raw)
                        gen_parsed_json = json.loads(gen_parsed_json)

                        self._log(f"JSON Output Valid: {True}")

                        if json.dumps(gt_parsed_json, sort_keys=True) == json.dumps(gen_parsed_json, sort_keys=True):
                            self._log("JSON Match: EXACT")
                        else:
                            self._log("JSON Match: DIFFERENCE (consider visual inspection)")
                    except json.JSONDecodeError as e:
                        self._log(f"JSON Output Valid: {False} - Error: {e}")
                        self._log(f"Raw generated text causing error: {generated_output_raw[:200]}...")
                    except Exception as e:
                        self._log(f"Error during JSON repair/parse: {e}")
                        self._log(f"Raw generated text causing error: {generated_output_raw[:200]}...")
            self.model.train()
            self.printed_steps.add(state.global_step)
