#!/usr/bin/env python3
"""
CLI Council - 基于 Karpathy 的三阶段机制 (优化版)
Stage 1: First opinions - 并行收集所有 LLM 的回答
Stage 2: Review - 每个 LLM 匿名评审所有回答并排名 (含自己)
Stage 3: Final response - Chairman 综合所有信息生成最终答案

参考: https://github.com/karpathy/llm-council
"""

import os
import subprocess
import sys
import re
import io
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from dataclasses import dataclass, field
from typing import Optional
from contextlib import redirect_stdout


@dataclass
class CliResult:
    """封装 CLI 调用结果"""
    name: str
    output: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class ReviewResult:
    """封装评审结果"""
    reviewer: str
    raw_text: str = ""
    parsed_ranking: list[str] = field(default_factory=list)  # ["Response A", "Response B", ...]


# CLI 配置 (按 Response A/B/C 顺序排列)
# 注意：Claude CLI 需要 PTY 环境，在无 TTY 环境下使用 script 命令模拟
CLIS = {
    "Codex": {          # Response A
        "cmd": ["codex", "exec", "--skip-git-repo-check", "--enable", "web_search_request", "-c", "model_reasoning_effort=\"high\""],
    },
    "Gemini": {         # Response B
        "cmd": ["gemini"],  # MCP 已在 ~/.gemini/settings.json 中禁用
    },
    "Claude Code": {    # Response C
        # 使用 script 命令模拟 PTY，解决无 TTY 环境下 Claude CLI 挂起的问题
        # use_script=True 表示需要用 script -c 包装命令
        "cmd": ["claude", "-p", "--permission-mode", "bypassPermissions", "--no-session-persistence"],
        "use_script": True,
    },
}

# Chairman 配置 (使用 Claude Code CLI)
CHAIRMAN_CMD = ["claude", "-p", "--permission-mode", "bypassPermissions", "--no-session-persistence"]
CHAIRMAN_USE_SCRIPT = True  # 在无 TTY 环境下需要用 script 模拟 PTY


def query_cli(name: str, config: dict, prompt: str, timeout: int = 300) -> CliResult:
    """调用单个 CLI 并返回结果"""
    try:
        import shlex
        base_cmd = config["cmd"]

        # 如果配置了 use_script，使用 script 命令模拟 PTY
        # 这是为了解决 Claude CLI 在无 TTY 环境下挂起的问题
        if config.get("use_script"):
            # 构建完整的命令字符串，然后用 script -q -c 包装
            full_cmd_str = " ".join(shlex.quote(arg) for arg in base_cmd) + " " + shlex.quote(prompt)
            cmd = ["script", "-q", "-c", full_cmd_str, "/dev/null"]
        else:
            cmd = base_cmd + [prompt]

        # 在临时目录中执行，避免 CLI 读取项目文件作为上下文
        with tempfile.TemporaryDirectory() as sandbox:
            # 设置环境变量，解决 Claude CLI 在无 TTY 环境下挂起的问题
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)    # 允许在 Claude Code 会话内嵌套调用 claude CLI
            env.update({
                "FORCE_COLOR": "0",        # 禁用颜色输出
                "CI": "true",              # 模拟 CI 环境，禁用交互式功能
                "TERM": "dumb",            # 使用简单终端类型
                "NODE_NO_READLINE": "1",   # 禁用 Node readline
            })
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout,
                cwd=sandbox,
                env=env,
                start_new_session=True,  # 创建新会话，避免 TTY 相关问题
            )
        output = result.stdout.strip() or result.stderr.strip()
        # 清理 ANSI 转义序列（script 命令模拟 PTY 时会产生这些控制字符）
        # 扩展正则以匹配更多 ANSI 序列格式，包括 \x1b[<u (restore cursor) 等
        output = re.sub(r'\x1b\[[0-9;?<>=]*[a-zA-Z]', '', output)
        output = re.sub(r'\x1b[PX^_].*?\x1b\\', '', output)  # 清理 DCS, SOS, PM, APC 序列
        output = re.sub(r'\x1b\][^\x07]*\x07', '', output)   # 清理 OSC 序列
        output = re.sub(r'\x1b[NO][\x20-\x7f]', '', output)  # 清理 SS2, SS3 序列
        # 清理其他常见控制字符
        output = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', output)

        # 检查返回码和输出内容是否包含错误
        if result.returncode != 0:
            return CliResult(name=name, error=f"退出码 {result.returncode}: {output[:200]}")

        # 检测常见的 API 错误模式（这些错误可能返回 0 退出码）
        error_patterns = [
            r'^API Error:',
            r'^Error:.*connection',
            r'^error:.*API',
            r'^Connection error',
            r'^Request failed',
        ]
        for pattern in error_patterns:
            if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                return CliResult(name=name, error=f"API 错误: {output[:200]}")

        return CliResult(name=name, output=output)
    except subprocess.TimeoutExpired:
        return CliResult(name=name, error="超时")
    except FileNotFoundError:
        return CliResult(name=name, error=f"未找到命令: {config['cmd'][0]}")
    except Exception as e:
        return CliResult(name=name, error=str(e))


def query_chairman(prompt: str, timeout: int = 300) -> str:
    """调用 Chairman (Claude Code CLI) 生成最终答案"""
    try:
        import shlex

        # 如果配置了 use_script，使用 script 命令模拟 PTY
        if CHAIRMAN_USE_SCRIPT:
            full_cmd_str = " ".join(shlex.quote(arg) for arg in CHAIRMAN_CMD) + " " + shlex.quote(prompt)
            cmd = ["script", "-q", "-c", full_cmd_str, "/dev/null"]
        else:
            cmd = CHAIRMAN_CMD + [prompt]

        # 在临时目录中执行，避免 CLI 读取项目文件作为上下文
        with tempfile.TemporaryDirectory() as sandbox:
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)    # 允许在 Claude Code 会话内嵌套调用 claude CLI
            env.update({
                "FORCE_COLOR": "0",
                "CI": "true",
                "TERM": "dumb",
                "NODE_NO_READLINE": "1",
            })
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout,
                cwd=sandbox,
                env=env,
                start_new_session=True,
            )
        output = result.stdout.strip() or result.stderr.strip()
        # 清理 ANSI 转义序列（与 query_cli 保持一致）
        output = re.sub(r'\x1b\[[0-9;?<>=]*[a-zA-Z]', '', output)
        output = re.sub(r'\x1b[PX^_].*?\x1b\\', '', output)  # 清理 DCS, SOS, PM, APC 序列
        output = re.sub(r'\x1b\][^\x07]*\x07', '', output)   # 清理 OSC 序列
        output = re.sub(r'\x1b[NO][\x20-\x7f]', '', output)  # 清理 SS2, SS3 序列
        output = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', output)

        # 检查返回码
        if result.returncode != 0:
            return f"[Chairman 错误: 退出码 {result.returncode}]"

        # 检测 API 错误模式
        error_patterns = [
            r'^API Error:',
            r'^Error:.*connection',
            r'^Connection error',
        ]
        for pattern in error_patterns:
            if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                return f"[Chairman API 错误: {output[:200]}]"

        return output
    except subprocess.TimeoutExpired:
        return "[Chairman 超时]"
    except FileNotFoundError:
        return "[未找到 claude 命令]"
    except Exception as e:
        return f"[Chairman 错误: {e}]"


def create_label_mapping(names: list[str]) -> dict[str, str]:
    """创建名称到匿名标签的映射 (Response A, B, C...)"""
    # 按名称排序以保证一致性，但标签是按顺序分配的
    labels = [f"Response {chr(65 + i)}" for i in range(len(names))]
    return {name: label for name, label in zip(names, labels)}


def parse_ranking_from_text(text: str, valid_labels: list[str]) -> list[str]:
    """
    从评审文本中解析排名
    查找 "FINAL RANKING:" 部分并提取排名列表

    Args:
        text: 完整的评审文本
        valid_labels: 有效的标签列表 ["Response A", "Response B", ...]

    Returns:
        排名列表，从最好到最差
    """
    rankings = []

    # 方法1: 查找 "FINAL RANKING:" 或 "最终排名" 部分
    final_ranking_match = re.search(
        r'(?:FINAL RANKING|最终排名|排名)[:\s：]*\n?(.*)',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if final_ranking_match:
        ranking_section = final_ranking_match.group(1)
        # 提取编号列表中的 Response X
        for line in ranking_section.split('\n'):
            for label in valid_labels:
                if label in line and label not in rankings:
                    rankings.append(label)
                    break

    # 方法2: 如果方法1失败，尝试在整个文本中按顺序查找 Response X
    if len(rankings) < len(valid_labels):
        # 查找所有 "Response X" 的出现位置
        pattern = r'Response [A-Z]'
        matches = re.findall(pattern, text)
        seen = set(rankings)
        for match in matches:
            if match in valid_labels and match not in seen:
                rankings.append(match)
                seen.add(match)

    return rankings


def calculate_aggregate_rankings(
    reviews: list[ReviewResult],
    label_to_model: dict[str, str]
) -> list[dict]:
    """
    计算聚合排名 - 基于平均位置

    Args:
        reviews: 所有评审结果
        label_to_model: 标签到模型名的映射

    Returns:
        排序后的聚合排名列表 [{"model": ..., "avg_position": ..., "votes": ...}, ...]
    """
    model_to_label = {v: k for k, v in label_to_model.items()}

    # 收集每个模型的所有排名位置
    position_sums: dict[str, float] = {model: 0.0 for model in label_to_model.values()}
    vote_counts: dict[str, int] = {model: 0 for model in label_to_model.values()}

    for review in reviews:
        if not review.parsed_ranking:
            continue
        for position, label in enumerate(review.parsed_ranking, start=1):
            model = label_to_model.get(label)
            if model:
                position_sums[model] += position
                vote_counts[model] += 1

    # 计算平均位置并排序
    results = []
    for model in label_to_model.values():
        votes = vote_counts[model]
        if votes > 0:
            avg_pos = position_sums[model] / votes
        else:
            avg_pos = float('inf')
        results.append({
            "model": model,
            "label": model_to_label[model],
            "avg_position": round(avg_pos, 2) if votes > 0 else None,
            "votes": votes
        })

    # 按平均位置排序（越小越好）
    results.sort(key=lambda x: x["avg_position"] if x["avg_position"] else float('inf'))
    return results


def stage1_first_opinions(question: str, verbose: bool = True, out_fn: Optional[callable] = None, return_output: bool = False) -> tuple[dict[str, CliResult], Optional[str]]:
    """
    Stage 1: 并行收集所有 LLM 的第一轮回答

    Returns:
        (results, output_text) - output_text 仅当 return_output=True 时有值
    """
    output_buffer = io.StringIO() if return_output else None

    def out(text: str = ""):
        if return_output:
            output_buffer.write(text + "\n")
        elif out_fn:
            out_fn(text)
        else:
            print(text)

    if verbose:
        out("\n**📝 Stage 1: First Opinions**\n")

    # 包装问题，添加格式约束（禁止表格）
    formatted_question = f"""{question}

【格式要求】请勿使用 Markdown 表格格式。如需展示对比或列表信息，请使用项目符号列表（如 - 或 •）或编号列表。"""

    results = {}
    with ThreadPoolExecutor(max_workers=len(CLIS)) as executor:
        futures = {
            executor.submit(query_cli, name, config, formatted_question): name
            for name, config in CLIS.items()
        }
        for future in as_completed(futures):
            cli_result = future.result()
            results[cli_result.name] = cli_result

    if verbose:
        # 按 CLIS 定义的顺序输出（Codex, Claude Code, Gemini）
        for name in CLIS.keys():
            if name in results:
                result = results[name]
                out(f"\n**{name}**\n")
                if result.success:
                    out(result.output)
                else:
                    out(f"[错误: {result.error}]")

    output_text = output_buffer.getvalue() if return_output else None
    return results, output_text


def stage2_review(
    question: str,
    results: dict[str, CliResult],
    verbose: bool = True,
    out_fn: Optional[callable] = None,
    return_output: bool = False
) -> tuple[dict[str, str], list[ReviewResult], Optional[str]]:
    """
    Stage 2: 每个 LLM 匿名评审所有回答（包括自己的）

    Returns:
        (label_to_model 映射, 评审结果列表, output_text)
    """
    output_buffer = io.StringIO() if return_output else None

    def out(text: str = ""):
        if return_output:
            output_buffer.write(text + "\n")
        elif out_fn:
            out_fn(text)
        else:
            print(text)

    if verbose:
        out("\n**🔍 Stage 2: Anonymous Peer Review**\n")

    # 过滤成功的结果，并保持 CLIS 定义的顺序
    valid_results = {name: results[name] for name in CLIS.keys() if name in results and results[name].success}
    if len(valid_results) < 2:
        out("\n[有效回答不足，跳过评审阶段]")
        return {}, [], output_buffer.getvalue() if return_output else None

    # 创建匿名映射（按 CLIS 顺序：Codex=A, Gemini=B, Claude Code=C）
    # model_to_label: {"Codex": "Response A", "Gemini": "Response B", "Claude Code": "Response C"}
    # label_to_model: {"Response A": "Codex", "Response B": "Gemini", "Response C": "Claude Code"}
    model_to_label = create_label_mapping(list(valid_results.keys()))
    label_to_model = {v: k for k, v in model_to_label.items()}
    valid_labels = list(label_to_model.keys())

    if verbose:
        out(f"\n🎭 匿名映射 (评审时模型看不到真实身份):")
        for model, label in model_to_label.items():
            out(f"   {label} = {model}")

    # 构建匿名回答文本
    responses_text = "\n\n".join(
        f"{model_to_label[name]}:\n{r.output}"
        for name, r in valid_results.items()
    )

    # 评审 prompt (中文)
    review_prompt = f"""你是一个评审员，需要评估以下问题的多个回答：

问题：{question}

以下是来自不同模型的回答（已匿名处理）：

{responses_text}

你的任务：
1. 首先，逐一评价每个回答。说明每个回答的优点和不足。
2. 然后，在回复的最后给出最终排名。

重要：你的最终排名必须严格按照以下格式：
- 以"最终排名："开头
- 然后按从最好到最差的顺序列出编号列表
- 每行格式：序号、点、空格、回答标签（如"1. Response A"）
- 排名部分不要添加任何其他解释文字

正确格式示例：

Response A 在X方面提供了详细信息，但遗漏了Y...

Response B 准确但在Z方面缺乏深度...

Response C 提供了最全面的答案...

最终排名：
1. Response C
2. Response A
3. Response B

请开始你的评价和排名："""

    # 并行执行评审
    review_results: list[ReviewResult] = []

    def do_review(reviewer_name: str) -> ReviewResult:
        result = query_cli(reviewer_name, CLIS[reviewer_name], review_prompt, timeout=300)
        if result.success:
            parsed = parse_ranking_from_text(result.output, valid_labels)
            return ReviewResult(
                reviewer=reviewer_name,
                raw_text=result.output,
                parsed_ranking=parsed
            )
        else:
            return ReviewResult(
                reviewer=reviewer_name,
                raw_text=f"[评审失败: {result.error}]",
                parsed_ranking=[]
            )

    with ThreadPoolExecutor(max_workers=len(valid_results)) as executor:
        # 按 CLIS 顺序提交任务，并保存 name -> future 的映射
        futures = {name: executor.submit(do_review, name) for name in valid_results.keys()}
        # 按 CLIS 顺序收集结果
        for name in CLIS.keys():
            if name in futures:
                review_results.append(futures[name].result())

    if verbose:
        # review_results 已经按 CLIS 顺序排列
        for review in review_results:
            out(f"\n**{review.reviewer} 的评审**\n")
            out(review.raw_text[:1500] + "..." if len(review.raw_text) > 1500 else review.raw_text)
            if review.parsed_ranking:
                out(f"\n📊 解析出的排名: {' > '.join(review.parsed_ranking)}")
            else:
                out(f"\n⚠️  未能解析出排名")

    output_text = output_buffer.getvalue() if return_output else None
    return label_to_model, review_results, output_text


def stage3_final_response(
    question: str,
    results: dict[str, CliResult],
    label_to_model: dict[str, str],
    reviews: list[ReviewResult],
    verbose: bool = True,
    out_fn: Optional[callable] = None,
    return_output: bool = False
) -> tuple[str, Optional[str]]:
    """
    Stage 3: Chairman 综合所有信息生成最终答案

    Returns:
        (final_answer, output_text)
    """
    output_buffer = io.StringIO() if return_output else None

    def out(text: str = ""):
        if return_output:
            output_buffer.write(text + "\n")
        elif out_fn:
            out_fn(text)
        else:
            print(text)

    if verbose:
        out("\n**⚖️  Stage 3: Chairman's Final Response**\n")

    valid_results = {name: r for name, r in results.items() if r.success}

    # 计算聚合排名
    aggregate_rankings = []
    if label_to_model and reviews:
        aggregate_rankings = calculate_aggregate_rankings(reviews, label_to_model)
        if verbose:
            out("\n📊 聚合排名 (基于平均位置):")
            for i, item in enumerate(aggregate_rankings, 1):
                pos_str = f"平均位置 {item['avg_position']}" if item['avg_position'] else "无投票"
                out(f"   {i}. {item['model']} ({pos_str}, {item['votes']}票)")
            out()  # 在聚合排名和最终答案之间加空行

    # 构建 Stage 1 回答
    stage1_text = "\n\n".join(
        f"Model: {name}\nResponse: {r.output}"
        for name, r in valid_results.items()
    )

    # 构建 Stage 2 评审摘要
    stage2_text = ""
    if reviews:
        stage2_parts = []
        for review in reviews:
            ranking_str = " > ".join(review.parsed_ranking) if review.parsed_ranking else "未提供排名"
            stage2_parts.append(f"Model: {review.reviewer}\nRanking: {ranking_str}")
        stage2_text = "\n\n".join(stage2_parts)

    # 构建聚合排名文本
    aggregate_text = ""
    if aggregate_rankings:
        lines = []
        for i, item in enumerate(aggregate_rankings, 1):
            pos_str = f"平均位置 {item['avg_position']}" if item['avg_position'] else "无投票"
            lines.append(f"{i}. {item['model']} ({pos_str})")
        aggregate_text = "\n聚合排名（基于平均位置）：\n" + "\n".join(lines)

    # Chairman prompt (中文)
    chairman_prompt = f"""你是 CLI Council 的主席。多个 AI 模型针对用户的问题给出了各自的回答，并且互相进行了匿名评审和排名。

用户问题：{question}

第一阶段 - 各模型回答：
{stage1_text}

第二阶段 - 互评排名：
{stage2_text}
{aggregate_text}

作为主席，你的任务是综合以上所有信息，给出一个最终的、高质量的答案。请考虑：
1. 每个回答中被指出的优点和不足
2. 互评排名中哪些回答被认为最准确
3. 各模型之间的共识与分歧
4. 聚合排名显示的整体模型表现

请取各家之长，纠正可能的错误，给出最佳答案。要求简洁而全面。

【格式要求】请勿使用 Markdown 表格格式。如需展示对比或列表信息，请使用项目符号列表（如 - 或 •）或编号列表。

最终答案："""

    final_answer = query_chairman(chairman_prompt, timeout=300)

    if verbose:
        out(final_answer)

    output_text = output_buffer.getvalue() if return_output else None
    return final_answer, output_text


def run_council(question: str, verbose: bool = True, skip_review: bool = False, return_output: bool = False) -> Optional[str]:
    """
    运行完整的 LLM Council 流程

    Args:
        question: 用户问题
        verbose: 是否输出详细信息
        skip_review: 是否跳过评审阶段
        return_output: 如果为 True，返回输出字符串而非打印

    Returns:
        如果 return_output=True，返回输出字符串；否则返回 None
    """
    output_buffer = io.StringIO() if return_output else None

    def out(text: str = ""):
        if return_output:
            output_buffer.write(text + "\n")
        else:
            print(text)

    if verbose:
        out(f"\n{'━'*40}")
        out("🏛️  CLI Council")
        out("━"*40)
        out(f"\n📋 问题: {question}\n")

    # Stage 1
    results, _ = stage1_first_opinions(question, verbose, out if return_output else None)

    # 统计（按 CLIS 顺序）
    successful = [name for name in CLIS.keys() if name in results and results[name].success]
    failed = [name for name in CLIS.keys() if name in results and not results[name].success]

    if verbose:
        out(f"\n📊 Stage 1 完成: ✅ {', '.join(successful) if successful else '无'}")
        if failed:
            out(f"                 ❌ {', '.join(failed)}")

    if len(successful) < 2:
        out("\n[有效回答不足2个，无法继续]")
        return output_buffer.getvalue() if return_output else None

    # Stage 2
    if skip_review:
        if verbose:
            out("\n⏭️  跳过 Stage 2 (Review)")
        label_to_model, reviews = {}, []
    else:
        label_to_model, reviews, _ = stage2_review(question, results, verbose, out if return_output else None)

    # Stage 3
    final_answer, _ = stage3_final_response(
        question, results, label_to_model, reviews, verbose, out if return_output else None
    )

    # 安静模式下只输出最终答案
    if not verbose:
        out(final_answer)
    else:
        out(f"\n{'━'*40}")
        out("✅ Council 流程完成")
        out("━"*40)

    return output_buffer.getvalue() if return_output else None


def main():
    parser = argparse.ArgumentParser(
        description="CLI Council - 多模型协作回答系统 (Karpathy 三阶段机制)"
    )
    parser.add_argument("question", nargs="?", help="要问的问题")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="安静模式，只显示关键输出")
    parser.add_argument("--skip-review", action="store_true",
                        help="跳过 Stage 2 评审阶段（更快但质量可能稍低）")
    args = parser.parse_args()

    # 获取问题
    if args.question:
        question = args.question
    else:
        print("请输入你的问题: ", end="", flush=True)
        question = input().strip()
        if not question:
            print("问题不能为空")
            sys.exit(1)

    verbose = not args.quiet
    run_council(question, verbose=verbose, skip_review=args.skip_review)


if __name__ == "__main__":
    main()
