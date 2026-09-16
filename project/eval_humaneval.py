# -*- coding: utf-8 -*-
"""eval_humaneval.py — HumanEval 自动解耦版（避免昇腾 NPU 与 os.fork 冲突）"""
import os, sys, json, time, argparse, gzip, subprocess, re
from transformers import AutoTokenizer

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--data", type=str, default="HumanEval.jsonl.gz", help="HumanEval 数据路径")
    parser.add_argument("--output", type=str, default="samples_humaneval.jsonl")
    return parser.parse_args()

def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model)
    model_name = os.path.basename(os.path.normpath(model_dir))
    
    data_path = args.data
    if not os.path.exists(data_path):
        candidates = [
            os.path.join(os.path.dirname(__file__), "HumanEval.jsonl.gz"),
            os.path.join(os.path.dirname(__file__), "data", "HumanEval.jsonl.gz"),
            os.path.join(os.path.dirname(__file__), "data", "benchmarks", "HumanEval.jsonl.gz"),
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    problems = {}
    with gzip.open(data_path, "rt", encoding="utf-8") if data_path.endswith(".gz") else open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            problems[d["task_id"]] = d
            
    task_ids = list(problems.keys())
    print(f"\n>>> 成功加载 HumanEval 题目: {len(task_ids)} 道")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    prompts = []
    for tid in task_ids:
        raw_prompt = problems[tid]["prompt"]
        dialog = [{"role": "user", "content": f"Complete the following Python function:\n```python\n{raw_prompt}\n```"}]
        formatted = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)
        
    print(f">>> 启动 vLLM 离线推理 ({len(prompts)} 题) ...")
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_dir, tensor_parallel_size=1, gpu_memory_utilization=0.5, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=512, top_p=1.0)
    outputs = llm.generate(prompts, sp)
    
    samples = []
    for i, out in enumerate(outputs):
        tid = task_ids[i]
        gen_text = out.outputs[0].text
        m = re.search(r"```(?:python)?\s*(.*?)\s*```", gen_text, re.DOTALL)
        code = m.group(1) if m else gen_text
        samples.append({"task_id": tid, "completion": code})
        
    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"\n[OK] 推理输出已写入 {args.output}")
    print(f">>> 启动隔离子进程进行官方严格判分 ...")
    
    eval_cmd = [
        sys.executable, "-c",
        f"from human_eval.evaluation import evaluate_functional_correctness; print(evaluate_functional_correctness('{args.output}'))"
    ]
    subprocess.run(eval_cmd, check=True)

if __name__ == "__main__":
    main()
