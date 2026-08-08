# V35 4B Offline GRPO

Transfer this entire directory to:

```text
/home/apulis-dev/userdata/VLMTSCS/training/verl/v35_offline_grpo
```

The launcher expects the merged SFT model and GRPO dataset at their standard
VLMTSCS server paths. Run a one-step smoke test from the VERL repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
ROLLOUT_N=2 \
bash v35_offline_grpo/run_v35_4b_offline_grpo.sh \
trainer.total_training_steps=1
```

Run the default full job with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash v35_offline_grpo/run_v35_4b_offline_grpo.sh
```
