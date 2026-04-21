CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
  --nproc_per_node=3 \
  --master_port=9833 \
  trainer_bary_ddp.py \
  --batchSize=3 \
  --nEpochs=65 \
  --pairnum=10000000 \
  --Sigma=10000 \
  --sigma=1 \
  --de_type derain dehaze deblur denoise_25 lowlight \
  --patch_size=144 \
  --type all \
  --backbone=BaryNet \
  --step=35 \
  --resume=checkpoint/model_allBaryNet128__58_1.0.pt \
  --num_sources=5

# AllWeather dataset training (3 GPUs, DDP)
# CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
#   --nproc_per_node=3 \
#   --master_port=9833 \
#   trainer_bary_ddp.py \
#   --allweather \
#   --allweather_dir=data/allweather/ \
#   --allweather_index=dataset-index.txt \
#   --batchSize=3 \
#   --nEpochs=65 \
#   --Sigma=10000 \
#   --sigma=1 \
#   --patch_size=128 \
#   --type allweather \
#   --backbone=BaryNet \
#   --step=35 \
#   --num_sources=3

# AllWeather dataset training (single GPU)
# python trainer_bary.py \
#   --allweather \
#   --allweather_dir=data/allweather/ \
#   --allweather_index=dataset-index.txt \
#   --batchSize=4 \
#   --nEpochs=200 \
#   --patch_size=128 \
#   --type allweather \
#   --backbone=BaryNet \
#   --num_sources=3