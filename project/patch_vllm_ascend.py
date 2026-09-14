# patch_vllm_ascend.py — 修 vllm-ascend 0.9.1rc1 在 verl 多 worker 下的通信组 bug
# 现象: verl 多卡 rollout 初始化时 AssertionError: assert self.cpu_group is not None
# 根因: init_ascend_model_parallel 按"引擎独占世界"视角建组——EP/ETP 组 ranks=[0]，
#       而 verl 的 N 个 worker 共享全局通信域(world=N)，全局 rank>0 的 worker 不在
#       [0] 组里 → cpu_group 停留 None → assert 崩（单卡 world=1 所以从不出事）
# 修法: 改用全局 world size 建 ETP 组(覆盖所有 rank)，EP 拆成单例组。
#       稠密模型(Qwen2.5)运行期不走 EP/ETP 通信，此改动安全；单卡行为不变。
# 用法: python3 patch_vllm_ascend.py （幂等，可重复跑；改的是容器层文件，每个新作业开头跑一次）
p = "/vllm-workspace/vllm-ascend/vllm_ascend/distributed/parallel_state.py"
s = open(p).read()

if "expert_tensor_parallel_size = world_size" in s:
    print("[patch_vllm_ascend] already patched, skip")
else:
    old = "    world_size = world_size or torch.distributed.get_world_size()"
    new = ("    world_size = torch.distributed.get_world_size()\n"
           "    expert_tensor_parallel_size = world_size\n"
           "    expert_parallel_size = 1")
    if old not in s:
        print("[patch_vllm_ascend] WARN: pattern not found —— vllm-ascend 版本与预期不符，"
              "可能新版本已自带修复，跳过补丁")
    else:
        open(p, "w").write(s.replace(old, new))
        print("[patch_vllm_ascend] PATCHED OK:", p)
