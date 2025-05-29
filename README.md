# Pattern-CoT

## Usage

1. Get zero-shot answers

```
python run_inference.py \
	--method zero_shot_cot \
	--dataset aqua \
	--model_name llama \
	--model_size 7b \
	--model_path PATH_TO_MODEL \
	--max_length_cot 256 2>&1 >log/aqua.log
```

2. Select pattern aware demos

```
python run_demos.py \
	--task aqua \
	--pred_file log/aqua.log \
	--demo_save_dir demos/aqua \
	--encoder PATH_TO_ENCODER
```

3. Enhance the CoT process
```
python run_inference.py \
	--method pattern_cot \
	--dataset aqua \
	--demo_path demos/aqua \
	--model_name llama \
	--model_size 7b \
	--model_path PATH_TO_MODEL \
	--max_length_cot 4096
```
