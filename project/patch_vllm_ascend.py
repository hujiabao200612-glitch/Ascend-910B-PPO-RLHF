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
