from transformers import AutoTokenizer, AutoModel, Seq2SeqTrainer, TrainingArguments, BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset, load_metric

if __name__ == '__main__':
    def map_to_encoder_decoder_inputs(batch):
        inputs = tokenizer(batch["article"],
                           padding="max_length",
                           truncation=True,
                           max_length=encoder_length)
        outputs = tokenizer(batch["highlights"],
                            padding="max_length",
                            truncation=True,
                            max_length=decoder_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["label"] = outputs.input_ids.copy()
        batch["label"] = [[-100 if token == tokenizer.pad_token_id else token
                            for token in label]
                           for label in batch["label"]]
        batch["decoder_attention_mask"] = outputs.attention_mask

        return batch


    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.eos_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str,
                                     rouge_types=["rouge1", "rouge2", "rougeL"])

        return format_rouge_output(rouge_output)


    def generate_summary(batch):
        inputs = tokenizer(batch["article"],
                           padding="max_length",
                           truncation=True,
                           max_length=model.config.max_length,
                           return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        outputs = model.generate(input_ids, attention_mask=attention_mask)

        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["pred"] = output_str

        return batch


    def format_rouge_output(rouge_output):
        return {"rouge1_precision": round(rouge_output["rouge1"].mid.precision, 4),
                "rouge1_recall":    round(rouge_output["rouge1"].mid.recall, 4),
                "rouge1_fmeasure":  round(rouge_output["rouge1"].mid.fmeasure, 4),
                "rouge2_precision": round(rouge_output["rouge2"].mid.precision, 4),
                "rouge2_recall":    round(rouge_output["rouge2"].mid.recall, 4),
                "rouge2_fmeasure":  round(rouge_output["rouge2"].mid.fmeasure, 4),
                "rougeL_precision": round(rouge_output["rougeL"].mid.precision, 4),
                "rougeL_recall":    round(rouge_output["rougeL"].mid.recall, 4),
                "rougeL_fmeasure":  round(rouge_output["rougeL"].mid.fmeasure, 4)}


    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)

    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    cnn_train_dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')
    cnn_val_dataset = load_dataset('cnn_dailymail', '3.0.0', split='validation')

    rouge = load_metric('rouge')

    # set decoding params
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    #model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 256
    model.config.min_length = 5
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 5

    encoder_length = 512
    decoder_length = 256
    batch_size = 2

    cnn_train_dataset = cnn_train_dataset.map(map_to_encoder_decoder_inputs,
                                              batched=True,
                                              batch_size=batch_size,
                                              remove_columns=["article", "highlights", "id"])
    cnn_train_dataset.set_format(type="torch",
                                 columns=["input_ids",
                                          "attention_mask",
                                          "decoder_attention_mask",
                                          "decoder_input_ids",
                                          "label"])

    # same for validation dataset
    cnn_val_dataset = cnn_val_dataset.map(map_to_encoder_decoder_inputs,
                                          batched=True,
                                          batch_size=batch_size,
                                          remove_columns=["article", "highlights", "id"])
    cnn_val_dataset.set_format(type="torch",
                               columns=["input_ids",
                                        "decoder_attention_mask",
                                        "attention_mask",
                                        "decoder_input_ids",
                                        "label"])

    training_args = TrainingArguments(output_dir="./cnn_dailymail",
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      do_train=True,
                                      do_eval=True,
                                      logging_steps=1000,
                                      save_steps=1000,
                                      eval_steps=1000,
                                      overwrite_output_dir=True,
                                      warmup_steps=2000,
                                      save_total_limit=3)

    # instantiate trainer
    trainer = Seq2SeqTrainer(model=model,
                             tokenizer=tokenizer,
                             args=training_args,
                             compute_metrics=compute_metrics,
                             train_dataset=cnn_train_dataset,
                             eval_dataset=cnn_val_dataset)

    # start training
    trainer.train()

    cnn_test_dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')

    results = cnn_test_dataset.map(generate_summary,
                                   batched=True,
                                   batch_size=batch_size,
                                   remove_columns=["article"])

    rouge_output = rouge.compute(predictions=results['pred'],
                                 references=results['highlights'],
                                 rouge_types=['rouge1', 'rouge2', 'rougeL'])

    print(format_rouge_output(rouge_output))
