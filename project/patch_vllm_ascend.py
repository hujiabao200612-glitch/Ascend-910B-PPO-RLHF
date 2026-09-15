# patch_vllm_ascend.py — 昇腾 8 卡环境补丁（幂等，训练前自动执行）
# 1. 修 vllm-ascend 0.9.1rc1 多 worker 通信组 bug (AssertionError: cpu_group is not None)
# 2. 修 verl 8 卡 rollout DataProto.concat 严格对比 timing 导致的崩溃 (AssertionError: Conflicting values for meta_info key 'timing')

import os
import re

# ----------------- 补丁 1: vllm-ascend 8 卡通信组 -----------------
p = "/vllm-workspace/vllm-ascend/vllm_ascend/distributed/parallel_state.py"
if os.path.exists(p):
    s = open(p).read()
    if "expert_tensor_parallel_size = world_size" in s:
        print("[patch_vllm_ascend] [1/2] vllm-ascend already patched, skip")
    else:
        old = "    world_size = world_size or torch.distributed.get_world_size()"
        new = ("    world_size = torch.distributed.get_world_size()\n"
               "    expert_tensor_parallel_size = world_size\n"
               "    expert_parallel_size = 1")
        if old not in s:
            print("[patch_vllm_ascend] [1/2] WARN: pattern not found in vllm-ascend, skip")
        else:
            open(p, "w").write(s.replace(old, new))
            print("[patch_vllm_ascend] [1/2] PATCHED OK: vllm-ascend communication group fixed")
else:
    print("[patch_vllm_ascend] [1/2] Note: vllm-ascend path not found (not in container root), skip")

# ----------------- 补丁 2: verl 多卡 rollout timing 对齐 -----------------
try:
    import verl
    verl_protocol_path = os.path.join(os.path.dirname(verl.__file__), "protocol.py")
    if os.path.exists(verl_protocol_path):
        content = open(verl_protocol_path, "r", encoding="utf-8").read()
        if 'if k != "timing":' in content or 'k == "timing"' in content:
            print("[patch_vllm_ascend] [2/2] verl protocol timing patch already applied, skip")
        else:
            pattern = r'([ \t]*)assert merged_meta_info\[k\] == v(.*)'
            match = re.search(pattern, content)
            if match:
                indent = match.group(1)
                rest = match.group(2)
                repl = f'{indent}if k != "timing":\n{indent}    assert merged_meta_info[k] == v{rest}'
                new_content = re.sub(pattern, repl, content, count=1)
                open(verl_protocol_path, "w", encoding="utf-8").write(new_content)
                print("[patch_vllm_ascend] [2/2] PATCHED OK: verl protocol timing fix applied to", verl_protocol_path)
            else:
                print("[patch_vllm_ascend] [2/2] WARN: target assert line not found in", verl_protocol_path)
except Exception as e:
    print(f"[patch_vllm_ascend] [2/2] WARN: failed to patch verl protocol: {e}")

# ----------------- 补丁 3: verl NaiveRewardManager CPU 64 线程并发加速 -----------------
try:
    import verl
    verl_naive_path = os.path.join(os.path.dirname(verl.__file__), "workers", "reward_manager", "naive.py")
    if os.path.exists(verl_naive_path):
        content = open(verl_naive_path, "r", encoding="utf-8").read()
        if "[Ascend-910B Parallel Patch]" in content:
            print("[patch_vllm_ascend] [3/3] verl reward manager parallel patch already applied, skip")
        else:
            pattern = r'([ \t]*)for i in range\(len\(data\)\):.*?(?=\n\1if return_dict:)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                indent = match.group(1)
                parallel_code = (
                    f"{indent}# [Ascend-910B Parallel Patch] 并发多线程沙箱判分加速\n"
                    f"{indent}items_to_score = []\n"
                    f"{indent}for i in range(len(data)):\n"
                    f"{indent}    data_item = data[i]\n"
                    f"{indent}    prompt_ids = data_item.batch[\"prompts\"]\n"
                    f"{indent}    prompt_length = prompt_ids.shape[-1]\n"
                    f"{indent}    valid_prompt_length = data_item.batch[\"attention_mask\"][:prompt_length].sum()\n"
                    f"{indent}    valid_prompt_ids = prompt_ids[-valid_prompt_length:]\n"
                    f"{indent}    response_ids = data_item.batch[\"responses\"]\n"
                    f"{indent}    valid_response_length = data_item.batch[\"attention_mask\"][prompt_length:].sum()\n"
                    f"{indent}    valid_response_ids = response_ids[:valid_response_length]\n"
                    f"{indent}    prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)\n"
                    f"{indent}    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)\n"
                    f"{indent}    ground_truth = data_item.non_tensor_batch[\"reward_model\"][\"ground_truth\"]\n"
                    f"{indent}    data_source = data_item.non_tensor_batch[self.reward_fn_key]\n"
                    f"{indent}    extra_info = data_item.non_tensor_batch.get(\"extra_info\", {{}})\n"
                    f"{indent}    num_turns = data_item.non_tensor_batch.get(\"__num_turns__\", None)\n"
                    f"{indent}    rollout_reward_scores = data_item.non_tensor_batch.get(\"reward_scores\", {{}})\n"
                    f"{indent}    extra_info[\"num_turns\"] = num_turns\n"
                    f"{indent}    extra_info[\"rollout_reward_scores\"] = rollout_reward_scores\n"
                    f"{indent}    items_to_score.append((i, valid_response_length, prompt_str, response_str, ground_truth, data_source, extra_info))\n"
                    f"\n"
                    f"{indent}from concurrent.futures import ThreadPoolExecutor\n"
                    f"{indent}import os\n"
                    f"{indent}max_workers = min(int(os.environ.get(\"VERL_REWARD_WORKERS\", 64)), len(items_to_score) or 1)\n"
                    f"\n"
                    f"{indent}def _eval_single(item):\n"
                    f"{indent}    idx, v_len, p_str, r_str, gt, ds, extra = item\n"
                    f"{indent}    score = self.compute_score(\n"
                    f"{indent}        data_source=ds,\n"
                    f"{indent}        solution_str=r_str,\n"
                    f"{indent}        ground_truth=gt,\n"
                    f"{indent}        extra_info=extra,\n"
                    f"{indent}    )\n"
                    f"{indent}    return idx, v_len, p_str, r_str, gt, ds, score\n"
                    f"\n"
                    f"{indent}with ThreadPoolExecutor(max_workers=max_workers) as executor:\n"
                    f"{indent}    scored_results = list(executor.map(_eval_single, items_to_score))\n"
                    f"\n"
                    f"{indent}for idx, valid_response_length, prompt_str, response_str, ground_truth, data_source, score in scored_results:\n"
                    f"{indent}    if isinstance(score, dict):\n"
                    f"{indent}        reward = score[\"score\"]\n"
                    f"{indent}        for key, value in score.items():\n"
                    f"{indent}            reward_extra_info[key].append(value)\n"
                    f"{indent}    else:\n"
                    f"{indent}        reward = score\n"
                    f"{indent}    reward_tensor[idx, valid_response_length - 1] = reward\n"
                    f"{indent}    if data_source not in already_print_data_sources:\n"
                    f"{indent}        already_print_data_sources[data_source] = 0\n"
                    f"{indent}    if already_print_data_sources[data_source] < self.num_examine:\n"
                    f"{indent}        already_print_data_sources[data_source] += 1\n"
                    f"{indent}        print(\"[prompt]\", prompt_str)\n"
                    f"{indent}        print(\"[response]\", response_str)\n"
                    f"{indent}        print(\"[ground_truth]\", ground_truth)\n"
                    f"{indent}        if isinstance(score, dict):\n"
                    f"{indent}            for key, value in score.items():\n"
                    f"{indent}                print(f\"[{{key}}]\", value)\n"
                    f"{indent}        else:\n"
                    f"{indent}            print(\"[score]\", score)"
                )
                new_content = content[:match.start()] + parallel_code + content[match.end():]
                open(verl_naive_path, "w", encoding="utf-8").write(new_content)
                print("[patch_vllm_ascend] [3/3] PATCHED OK: verl NaiveRewardManager 64-thread parallelization applied to", verl_naive_path)
            else:
                print("[patch_vllm_ascend] [3/3] WARN: target loop pattern not found in", verl_naive_path)
    else:
        print("[patch_vllm_ascend] [3/3] Note: verl naive reward manager path not found, skip")
except Exception as e:
    print(f"[patch_vllm_ascend] [3/3] WARN: failed to patch verl reward manager: {e}")

