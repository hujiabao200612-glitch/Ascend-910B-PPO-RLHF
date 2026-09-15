# -*- coding: utf-8 -*-
"""plot_d4_curves.py — 解析 D4 全量 71 步训练日志并生成学术级 4 面板可视化图表"""

import os
import re
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_log():
    logs = glob.glob("train_d4_*.log") + glob.glob("project/train_d4_*.log")
    if not logs:
        logs = glob.glob("*.log")
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def parse_log_file(log_path):
    steps = []
    rewards = []
    vf_losses = []
    entropies = []
    resp_lens = []
    timing_rewards = []
    timing_steps = []
    throughputs = []

    pattern = re.compile(r'step:(\d+)\s+-\s+(.+)')

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            step_num = int(m.group(1))
            kv_str = m.group(2)
            kvs = {}
            for item in kv_str.split(' - '):
                parts = item.strip().split(':')
                if len(parts) == 2:
                    try:
                        kvs[parts[0].strip()] = float(parts[1].strip())
                    except ValueError:
                        pass
            
            if 'critic/rewards/mean' in kvs or 'critic/score/mean' in kvs:
                steps.append(step_num)
                rewards.append(kvs.get('critic/rewards/mean', kvs.get('critic/score/mean', 0.0)))
                vf_losses.append(kvs.get('critic/vf_loss', 0.0))
                entropies.append(kvs.get('actor/entropy', 0.0))
                resp_lens.append(kvs.get('response_length/mean', 0.0))
                timing_rewards.append(kvs.get('timing_s/reward', 0.0))
                timing_steps.append(kvs.get('timing_s/step', 0.0))
                throughputs.append(kvs.get('perf/throughput', 0.0))

    return {
        'steps': steps,
        'rewards': rewards,
        'vf_losses': vf_losses,
        'entropies': entropies,
        'resp_lens': resp_lens,
        'timing_rewards': timing_rewards,
        'timing_steps': timing_steps,
        'throughputs': throughputs,
    }


def plot_curves(data, out_path="d4_training_curves.png"):
    steps = data['steps']
    if not steps:
        print("未在日志中解析到 step 数据！")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200)
    fig.patch.set_facecolor('#fafbfc')

    # 1. 奖励变化趋势
    ax1 = axes[0, 0]
    ax1.set_facecolor('#ffffff')
    ax1.plot(steps, data['rewards'], color='#0284c7', linewidth=2.2, label='平均奖励 (Reward Mean)')
    if len(steps) > 5:
        z = np.polyfit(steps, data['rewards'], 2)
        p = np.poly1d(z)
        ax1.plot(steps, p(steps), "r--", alpha=0.75, linewidth=1.6, label='收敛趋势线')
    ax1.set_title('图 1: 奖励得分演进 (Reward Convergence)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('训练步数 (Global Steps)', fontsize=10)
    ax1.set_ylabel('平均奖励分 (0.0 ~ 1.0)', fontsize=10)
    ax1.set_ylim(0.60, 0.95)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.annotate(f'起始: {data["rewards"][0]:.4f}', (steps[0], data["rewards"][0]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='gray'))
    max_idx = np.argmax(data['rewards'])
    ax1.annotate(f'峰值: {data["rewards"][max_idx]:.4f} (step {steps[max_idx]})', 
                 (steps[max_idx], data['rewards'][max_idx]),
                 textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9,
                 fontweight='bold', color='#dc2626', arrowprops=dict(arrowstyle='->', color='#dc2626'))

    # 2. Critic Value Loss
    ax2 = axes[0, 1]
    ax2.set_facecolor('#ffffff')
    ax2.plot(steps, data['vf_losses'], color='#ea580c', linewidth=2.2, label='Critic Value Loss (均方误差)')
    ax2.set_title('图 2: Critic 价值网络损失收敛 (VF Loss)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('训练步数 (Global Steps)', fontsize=10)
    ax2.set_ylabel('Critic Loss (对数尺度)', fontsize=10)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.5, which='both')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.annotate(f'从 {data["vf_losses"][0]:.2f} 暴降', (steps[0], data["vf_losses"][0]),
                 textcoords="offset points", xytext=(20, 0), ha='left', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='gray'))
    min_idx = np.argmin(data['vf_losses'])
    drop_pct = (1.0 - data["vf_losses"][min_idx] / data["vf_losses"][0]) * 100
    ax2.annotate(f'收敛至 {data["vf_losses"][min_idx]:.4f} (-{drop_pct:.1f}%)', 
                 (steps[min_idx], data['vf_losses'][min_idx]),
                 textcoords="offset points", xytext=(-20, 15), ha='center', fontsize=9,
                 fontweight='bold', color='#ea580c', arrowprops=dict(arrowstyle='->', color='#ea580c'))

    # 3. 策略熵与生成长度
    ax3 = axes[1, 0]
    ax3.set_facecolor('#ffffff')
    ax3_twin = ax3.twinx()
    l1 = ax3.plot(steps, data['entropies'], color='#7c3aed', linewidth=2.0, label='策略熵 (Actor Entropy)')
    l2 = ax3_twin.plot(steps, data['resp_lens'], color='#059669', linewidth=2.0, linestyle='-.', label='平均生成长度 (Tokens)')
    ax3.set_title('图 3: 策略熵与代码生成长度 (Entropy & Response Length)', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('训练步数 (Global Steps)', fontsize=10)
    ax3.set_ylabel('策略熵 (Policy Entropy)', color='#7c3aed', fontsize=10)
    ax3_twin.set_ylabel('平均长度 (Tokens)', color='#059669', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    lines = l1 + l2
    ax3.legend(lines, [l.get_label() for l in lines], loc='upper right', framealpha=0.9)

    # 4. 耗时与吞吐 (单步总耗时 vs 沙箱耗时)
    ax4 = axes[1, 1]
    ax4.set_facecolor('#ffffff')
    ax4.plot(steps, data['timing_steps'], color='#2563eb', linewidth=2.0, label='单步总耗时 (~47s)')
    ax4.plot(steps, data['timing_rewards'], color='#d97706', linewidth=2.0, label='64并发沙箱打分耗时 (~3-5s)')
    ax4.set_title('图 4: 单步耗时与沙箱并发提速 (Step & Reward Timing)', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel('训练步数 (Global Steps)', fontsize=10)
    ax4.set_ylabel('耗时 (秒)', fontsize=10)
    ax4.set_ylim(0, max(data['timing_steps']) * 1.25)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='center right', framealpha=0.9)
    ax4.annotate(f'沙箱并发稳定在 ~{np.mean(data["timing_rewards"]):.1f}s', (steps[len(steps)//2], np.mean(data["timing_rewards"])),
                 textcoords="offset points", xytext=(0, 20), ha='center', fontsize=9,
                 fontweight='bold', color='#d97706', arrowprops=dict(arrowstyle='->', color='#d97706'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[OK] 训练指标曲线图已成功导出至: {out_path}")


if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else find_latest_log()
    if not log_file or not os.path.exists(log_file):
        print("错误: 未找到训练日志文件！请提供日志路径，如 python plot_d4_curves.py train_d4_0915_1525.log")
        sys.exit(1)
    print(f"正在解析训练日志: {log_file} ...")
    metrics = parse_log_file(log_file)
    print(f"共解析到 {len(metrics['steps'])} 步训练数据。")
    plot_curves(metrics)
