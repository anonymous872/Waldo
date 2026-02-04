export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3,4

python main_field.py --model_name "Qwen/Qwen2-VL-7B-Instruct" --benchmark "CharXiv" --output_path "./results_field"