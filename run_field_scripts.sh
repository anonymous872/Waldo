BENCHMARKS=("CharXiv" "MMVet" "MathVerse" "MMEval" "MMStar")

MODELS=(
  "Qwen/Qwen2-VL-2B-Instruct"
  "Qwen/Qwen2-VL-7B-Instruct"
  "Qwen/Qwen2-VL-72B-Instruct"
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-32B-Instruct"
  "Qwen/Qwen2.5-VL-72B-Instruct"
  "Qwen/Qwen3-VL-2B-Instruct"
  "Qwen/Qwen3-VL-4B-Instruct"
  "Qwen/Qwen3-VL-8B-Instruct"
  "Qwen/Qwen3-VL-32B-Instruct"
)

for BENCH in "${BENCHMARKS[@]}"; do
  for MODEL in "${MODELS[@]}"; do

    echo "Running model=${MODEL}, benchmark=${BENCH}"

    python main_field.py \
      --model_name "$MODEL" \
      --benchmark "$BENCH" \
      --output_path "./results_field/${MODEL//\//_}/${BENCH}"

  done
done