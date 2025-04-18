### 🔧 Example Configuration

```python
sd = 1
a = 1
unbalance_value = 0.0
dlist = 0.3  # data ratio
C = 0.1      # active rate
task_idx = "fashion_mnist"  # only one task for this toy example
iid = "noniid noniid noniid noniid noniid"
client_n = 40
class_ratio = 0.3
```

### 🚀 Run Command

```bash
python main2.py \
  --powerfulCNN \
  --skipOS \
  --venn_list 0.9 0.1 0.0 \
  --freshness \
  --fairness notfair \
  --data_ratio $d \
  --unbalance $uv 1.0 \
  --alpha $a \
  --notes "${task_idx}_class${class_ratio}c${C}uvNo_a5.0_random_${sd}" \
  --optimal_sampling \
  --alpha_loss \
  --C $C \
  --num_clients $client_n \
  --class_ratio $class_ratio $class_ratio $class_ratio $class_ratio $class_ratio \
  --iid_type "$iid" \
  --task_type "$task_idx" \
  --algo_type proposed \
  --seed $sd \
  --cpumodel \
  --local_epochs 5 5 5 5 5 \
  --round_num 150 \
  --insist
```
