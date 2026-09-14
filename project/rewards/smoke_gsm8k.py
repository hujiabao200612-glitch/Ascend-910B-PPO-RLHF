# smoke_gsm8k.py — 冒烟用奖励函数 wrapper
# 作用：兼容 verl 0.6.1 naive 奖励管理器的传参方式（多了 data_source/extra_info）
# 后续的自设计奖励（三层结构：内核+整形+惩罚）就演化自这个文件
try:
    from verl.utils.reward_score.gsm8k import compute_score as _gsm8k_score
except ImportError:
    import sys
    sys.path.insert(0, '/usr/local/python3.10.17/lib/python3.10/site-packages')
    from verl.utils.reward_score.gsm8k import compute_score as _gsm8k_score

def compute_score(data_source=None, solution_str=None, ground_truth=None,
                  extra_info=None, **kwargs):
    gt = ground_truth
    if isinstance(gt, dict):                     # 兼容 dict/str 两种 ground_truth 格式
        gt = gt.get("ground_truth", str(gt))
    return _gsm8k_score(solution_str, gt)
