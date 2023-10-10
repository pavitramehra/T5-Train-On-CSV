from huggingface_hub import notebook_login
notebook_login()

import datasets

from t5_tokenizer_model import SentencePieceUnigramTokenizer


vocab_size = 32_000
input_sentence_size = None

# Initialize a dataset
dataset = datasets.load_dataset("finalContent", name="unshuffled_deduplicated_no", split="train")

tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
    batch_length = 100
    for i in range(0, input_sentence_size, batch_length):
        yield dataset[i: i + batch_length]["text"]


# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(input_sentence_size=input_sentence_size),
    vocab_size=vocab_size,
    show_progress=True,
)

# Save files to disk
tokenizer.save("./norwegian-t5-base/tokenizer.json")

python run_t5_mlm_flax.py \
	--output_dir="/content/drive/MyDrive/privateT1" \
	--model_type="t5" \
	--config_name="/content/drive/MyDrive/privateT5" \
	--tokenizer_name="/content/drive/MyDrive/privateT5" \
	--dataset_name="Pavitra05/finalContent" \
	--dataset_config_name="unshuffled_deduplicated_no" \
	--max_seq_length="512" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500" \
	--push_to_hub