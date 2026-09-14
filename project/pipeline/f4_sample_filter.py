# -*- coding: utf-8 -*-
"""f4_sample_filter.py —— 筛 4：7B 基座预采样与难度分级（任务书 §5.7 + 数据清单 §4）。

职责
    1. 读取筛 3 产出的高质量题目池（data/step3_verified_pool.jsonl）；
    2. 使用 Qwen2.5-7B-Instruct 基座模型，对每道题进行 8 次采样（temp 1.0, max_tokens 768）；
    3. 支持多卡数据并行（Data Parallelism，支持 1~8 卡），8 张 910B 同时开跑，提速 8 倍；
    4. 对采样出的 8 个回答调用沙箱判分，计算每道题的通过率 pass_rate = 做对次数 / 8；
    5. 难度分级过滤：
       - pass_rate > 0.9（全对/极易）：太简单，模型无梯度提升空间，淘汰到 step4_rejects.jsonl (reason: too_easy)；
       - pass_rate < 0.1（全错/极难）：太难，模型无正向反馈引导，淘汰到 step4_rejects.jsonl (reason: too_hard)；
       - 0.1 <= pass_rate <= 0.9：黄金难度题目，保留！
    6. 从保留题目中随机隔离 200 条作为考试集 heldout.jsonl（绝不进入训练池）；
    7. 剩余题目保存为最终训练池 rl_pool.jsonl（目标 2,000 ~ 4,000 条）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from score import score_kernel


def evaluate_question_samples(record: Dict, responses: List[str]) -> Tuple[float, int]:
    """对一道题目的 8 个采样结果跑沙箱判分，返回 (pass_rate, passed_trials_count)。"""
    test_code = record.get("test", "")
    if not test_code or not responses:
        return 0.0, 0

    passed_count = 0
    for resp in responses:
        passed, total, detail = score_kernel(resp, test_code, timeout_s=10)
        # 严格通过标准：测试全部通过且未超时且提取出代码
        if total > 0 and passed == total and not detail.get("timed_out", False):
            passed_count += 1

    pass_rate = passed_count / float(len(responses))
    return pass_rate, passed_count


def _worker_sample_and_score(
    gpu_id: int,
    records_chunk: List[Dict],
    model_path: str,
    worker_out_path: str,
    n_samples: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 768,
    gpu_memory_utilization: float = 0.6,
    batch_size: int = 50,
) -> None:
    """子进程工作函数：单张卡加载 vLLM 实例，分批采样并实时过沙箱判分，直写本地文件。"""
    # 绑定特定 NPU/GPU 显卡
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print(f"[Worker-{gpu_id}][ERROR] 未安装 vLLM，请在 verl_env 环境下运行！", file=sys.stderr)
        return

    print(f"[Worker-{gpu_id}] 正在卡 {gpu_id} 上加载模型: {model_path} ...", flush=True)
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    except Exception as exc:
        print(f"[Worker-{gpu_id}][ERROR] 模型加载失败: {exc}", file=sys.stderr)
        return

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    total_chunks = len(records_chunk)

    # 自动断点检测：统计已完成题数，实现无缝断点续跑
    already_done = 0
    if os.path.exists(worker_out_path):
        with open(worker_out_path, "r", encoding="utf-8") as fr:
            for line in fr:
                if line.strip():
                    already_done += 1

    if already_done >= total_chunks:
        print(f"[Worker-{gpu_id}] 本卡所有 {total_chunks} 题此前已全部完成，无需重跑！", flush=True)
        return

    if already_done > 0:
        print(f"[Worker-{gpu_id}] ⚡ 成功检测到断点！已完成 {already_done}/{total_chunks} 题，从第 {already_done + 1} 题继续运行...", flush=True)
    else:
        print(f"[Worker-{gpu_id}] 就绪！开始分批处理共 {total_chunks} 道题目 (batch_size={batch_size}) ...", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(worker_out_path)), exist_ok=True)
    open_mode = "a" if already_done > 0 else "w"
    with open(worker_out_path, open_mode, encoding="utf-8") as fw:
        for b_start in range(already_done, total_chunks, batch_size):
            b_chunk = records_chunk[b_start : b_start + batch_size]
            b_prompts = [rec["prompt"] for rec in b_chunk]
            try:
                b_outputs = llm.generate(b_prompts, sampling_params)
            except Exception as exc:
                print(f"[Worker-{gpu_id}][ERROR] generate 异常: {exc}", file=sys.stderr)
                continue

            for rec, out in zip(b_chunk, b_outputs):
                sample_texts = [output.text for output in out.outputs]
                pass_rate, passed_count = evaluate_question_samples(rec, sample_texts)
                result_item = dict(rec)
                result_item["sample_pass_rate"] = round(pass_rate, 4)
                result_item["sample_passed_count"] = passed_count
                result_item["sample_total"] = len(sample_texts)
                fw.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            fw.flush()
            print(f"[Worker-{gpu_id}] 进度: {min(b_start + batch_size, total_chunks)}/{total_chunks} 完成", flush=True)

    print(f"[Worker-{gpu_id}] 全部完成！产出写入: {worker_out_path}", flush=True)


def run_filter4_parallel(
    input_path: str,
    rl_pool_path: str,
    heldout_path: str,
    rejects_path: str,
    model_path: str,
    gpus: int = 8,
    max_problems: Optional[int] = None,
    batch_size: int = 50,
    n_samples: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 768,
    pass_rate_min: float = 0.1,
    pass_rate_max: float = 0.9,
    n_heldout: int = 200,
    gpu_memory_utilization: float = 0.6,
    seed: int = 42,
) -> None:
    """筛 4 主执行流程：多卡数据并行采样 + 难度分级。"""
    random.seed(seed)
    print(f"[f4_sample_filter] 正在读取输入文件: {input_path} ...")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_problems is not None and max_problems > 0:
        records = records[:max_problems]
        print(f"[f4_sample_filter] 开启 --max_problems 截断，本次采样前 {len(records)} 道题目")

    # ⚡ 自动扫描已有产出，实现断点增量续跑（跳过已处理题目，避免重复计算）
    already_done_ids = set()
    for existing_file in [rl_pool_path, heldout_path, rejects_path]:
        if os.path.exists(existing_file):
            with open(existing_file, "r", encoding="utf-8") as fe:
                for line in fe:
                    line = line.strip()
                    if line:
                        try:
                            qid = json.loads(line).get("question_id")
                            if qid:
                                already_done_ids.add(qid)
                        except Exception:
                            pass

    if already_done_ids:
        orig_count = len(records)
        records = [r for r in records if r.get("question_id") not in already_done_ids]
        print(f"[f4_sample_filter] ⚡ 发现已完成题目 {len(already_done_ids)} 道！自动开启增量续跑模式，剩余待处理题目: {len(records)}/{orig_count} 道。")

    if not records:
        print("[f4_sample_filter] 🎉 所有题目已全部完成，无需重跑！")
        return

    total_records = len(records)
    print(f"[f4_sample_filter] 本次待处理 {total_records} 道题目，分配至 {gpus} 张显卡并行处理...")

    # 拆分成 chunks
    chunks = [records[i::gpus] for i in range(gpus)]

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    processes = []
    worker_files = []

    out_dir = os.path.dirname(os.path.abspath(rl_pool_path))
    for gpu_id in range(gpus):
        worker_out = os.path.join(out_dir, f"_tmp_f4_worker_{gpu_id}.jsonl")
        worker_files.append(worker_out)
        p = mp.Process(
            target=_worker_sample_and_score,
            args=(
                gpu_id,
                chunks[gpu_id],
                model_path,
                worker_out,
                n_samples,
                temperature,
                max_tokens,
                gpu_memory_utilization,
                batch_size,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 汇总所有 worker 的结果
    all_results = []
    for wf in worker_files:
        if os.path.exists(wf):
            with open(wf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_results.append(json.loads(line))
            try:
                os.remove(wf)
            except OSError:
                pass

    print(f"[f4_sample_filter] 本次新增 {len(all_results)} 道题目采样与判分已完成，正在分流过滤...")

    kept_candidates = []
    too_easy_count = 0
    too_hard_count = 0

    os.makedirs(os.path.dirname(os.path.abspath(rl_pool_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(heldout_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(rejects_path)), exist_ok=True)

    # 追加写入淘汰集
    with open(rejects_path, "a", encoding="utf-8") as frej:
        for item in all_results:
            pr = item.get("sample_pass_rate", 0.0)
            if pr > pass_rate_max:
                too_easy_count += 1
                item["filter_reason"] = "too_easy"
                frej.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif pr < pass_rate_min:
                too_hard_count += 1
                item["filter_reason"] = "too_hard"
                frej.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                kept_candidates.append(item)

    print(f"[f4_sample_filter] 本轮分级: 新增黄金难度 {len(kept_candidates)} | 极易淘汰 {too_easy_count} | 极难淘汰 {too_hard_count}")

    # 检查 heldout 是否已满
    current_heldout_count = 0
    if os.path.exists(heldout_path):
        with open(heldout_path, "r", encoding="utf-8") as fh:
            current_heldout_count = sum(1 for line in fh if line.strip())

    needed_heldout = max(0, n_heldout - current_heldout_count)
    random.shuffle(kept_candidates)
    heldout_items = kept_candidates[:needed_heldout]
    rl_pool_items = kept_candidates[needed_heldout:]

    if heldout_items:
        with open(heldout_path, "a", encoding="utf-8") as fheld:
            for item in heldout_items:
                fheld.write(json.dumps(item, ensure_ascii=False) + "\n")

    if rl_pool_items:
        with open(rl_pool_path, "a", encoding="utf-8") as frl:
            for item in rl_pool_items:
                frl.write(json.dumps(item, ensure_ascii=False) + "\n")

    final_rl_count = sum(1 for _ in open(rl_pool_path, encoding="utf-8")) if os.path.exists(rl_pool_path) else 0
    final_held_count = sum(1 for _ in open(heldout_path, encoding="utf-8")) if os.path.exists(heldout_path) else 0
    final_rej_count = sum(1 for _ in open(rejects_path, encoding="utf-8")) if os.path.exists(rejects_path) else 0

    print(f"\n==================== [筛 4 累计成果清单] ====================")
    print(f"本次新处理题数: {len(all_results)}")
    print(f"当前累计强化学习训练池 (rl_pool.jsonl): {final_rl_count} 道")
    print(f"当前累计独立评测集 (heldout.jsonl):     {final_held_count} 道")
    print(f"当前累计淘汰集 (step4_rejects.jsonl):   {final_rej_count} 道")
    print(f"总计已处理: {final_rl_count + final_held_count + final_rej_count} 道题目")
    print(f"===========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="筛 4：7B 预采样与难度分级（支持 1~8 卡数据并行加速）")
    parser.add_argument("--input", default="data/step3_verified_pool.jsonl", help="筛 3 输出路径")
    parser.add_argument("--rl_pool", default="data/rl_pool.jsonl", help="最终强化学习池输出路径")
    parser.add_argument("--heldout", default="data/heldout.jsonl", help="独立评测集输出路径 (200题)")
    parser.add_argument("--rejects", default="data/step4_rejects.jsonl", help="淘汰集输出路径")
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct", help="7B 模型路径")
    parser.add_argument("--gpus", type=int, default=8, help="并行卡数（默认 8 卡，可传 1~8）")
    parser.add_argument("--max_problems", type=int, default=None, help="最大采样题数（默认全部，可传如 6000 快速收敛 2~3k 黄金池）")
    parser.add_argument("--batch_size", type=int, default=50, help="每批次推理题数（默认 50）")
    parser.add_argument("--n_samples", type=int, default=8, help="每题采样次数（默认 8）")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度（默认 1.0）")
    parser.add_argument("--max_tokens", type=int, default=768, help="最大生成 token 数")
    parser.add_argument("--pass_rate_min", type=float, default=0.1, help="最低保留通过率")
    parser.add_argument("--pass_rate_max", type=float, default=0.9, help="最高保留通过率")
    parser.add_argument("--n_heldout", type=int, default=200, help="留出评测题数")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6, help="显存使用上限比例（默认 0.6 安全值）")

    args = parser.parse_args()

    run_filter4_parallel(
        input_path=args.input,
        rl_pool_path=args.rl_pool,
        heldout_path=args.heldout,
        rejects_path=args.rejects,
        model_path=args.model,
        gpus=args.gpus,
        max_problems=args.max_problems,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        pass_rate_min=args.pass_rate_min,
        pass_rate_max=args.pass_rate_max,
        n_heldout=args.n_heldout,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
