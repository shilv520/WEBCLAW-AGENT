"""
阶段1 测试脚本 - Boss直聘求职搜索测试

运行方式: cd d:/webclaw-agent && .venv/Scripts/python.exe tests/test_week1.py

测试任务:
1. 打开Boss直聘
2. 搜索 "全国 大模型应用开发 langchain RAG"
3. 筛选条件：全职、本科、22-35岁、1-3年经验、10-20K
4. 查看前8个岗位，记录技术要求
5. 筛选含Agent技能的岗位，收藏5个
6. 总结核心技术需求
7. 对比自身技能，列出短板
8. 生成技能提升清单
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.playwright_browser import PlaywrightBrowser
from loguru import logger


async def boss_zhipin_test():
    """Boss直聘求职搜索完整测试"""
    print("=" * 60)
    print("Boss直聘求职搜索测试")
    print("=" * 60)

    browser = PlaywrightBrowser(headless=False)

    # 收集的数据
    job_data = []
    agent_jobs = []
    skill_summary = {}

    try:
        await browser.start()

        # ========================================
        # 步骤1: 打开Boss直聘
        # ========================================
        print("\n[步骤1] 打开Boss直聘...")
        result = await browser.open_url("https://www.zhipin.com/")
        print(f"  结果: {result}")
        await browser.wait(3)

        # ========================================
        # 步骤2: 搜索职位
        # ========================================
        print("\n[步骤2] 搜索 '大模型应用开发 langchain RAG'...")

        # Boss直聘搜索框选择器
        search_selector = ".ipt-search"  # 搜索输入框

        # 输入搜索词
        result = await browser.type_text(search_selector, "大模型应用开发 langchain RAG")
        print(f"  输入结果: {result}")
        await browser.wait(1)

        # 点击搜索按钮
        result = await browser.click(".btn-search")
        print(f"  搜索结果: {result}")
        await browser.wait(3)

        # ========================================
        # 步骤3: 设置筛选条件
        # ========================================
        print("\n[步骤3] 设置筛选条件...")

        # 点击筛选按钮区域
        # 注意：Boss直聘的筛选UI可能需要逐个点击设置

        # 尝试点击"全职"筛选
        try:
            await browser.click("text=全职")
            print("  已选择: 全职")
            await browser.wait(0.5)
        except:
            print("  全职筛选可能已默认选中")

        # 尝试设置学历筛选
        try:
            await browser.click("text=学历要求")
            await browser.wait(0.5)
            await browser.click("text=本科")
            print("  已选择: 本科")
            await browser.wait(0.5)
        except:
            print("  学历筛选跳过")

        # 尝试设置经验筛选
        try:
            await browser.click("text=经验要求")
            await browser.wait(0.5)
            await browser.click("text=1-3年")
            print("  已选择: 1-3年经验")
            await browser.wait(0.5)
        except:
            print("  经验筛选跳过")

        # 尝试设置薪资筛选
        try:
            await browser.click("text=薪资要求")
            await browser.wait(0.5)
            await browser.click("text=10-20K")
            print("  已选择: 10-20K薪资")
            await browser.wait(0.5)
        except:
            print("  赔资筛选跳过")

        await browser.wait(2)
        await browser.screenshot("filtered_jobs")

        # ========================================
        # 步骤4: 查看前8个岗位
        # ========================================
        print("\n[步骤4] 查看前8个岗位描述...")

        # 获取岗位列表
        job_items = await browser.find_elements(".job-list-item")
        print(f"  找到 {len(job_items)} 个岗位")

        # 逐个查看岗位详情
        for i in range(min(8, len(job_items))):
            print(f"\n  --- 岗位 {i+1} ---")

            try:
                # 点击岗位进入详情
                await browser.click(f".job-list-item:nth-child({i+1})")
                await browser.wait(2)

                # 获取岗位标题
                title = await browser.extract_text(".job-title")
                print(f"  标题: {title[:50]}")

                # 获取岗位详情/技术要求
                detail = await browser.extract_text(".job-detail")
                print(f"  详情摘要: {detail[:200]}...")

                # 记录数据
                job_info = {
                    "index": i + 1,
                    "title": title,
                    "detail": detail,
                    "has_agent": "agent" in detail.lower() or "Agent" in detail or "智能体" in detail
                }
                job_data.append(job_info)

                # 检查是否含Agent技能
                if job_info["has_agent"]:
                    print("  ✓ 含Agent相关技能")
                    agent_jobs.append(job_info)
                else:
                    print("  - 无Agent技能要求")

                # 截图记录
                await browser.screenshot(f"job_detail_{i+1}")

                # 返回列表
                await browser.click(".back-btn")
                await browser.wait(1)

            except Exception as e:
                print(f"  查看岗位{i+1}失败: {e}")
                continue

        # ========================================
        # 步骤5: 筛选含Agent岗位，收藏5个
        # ========================================
        print("\n[步骤5] 筛选含Agent岗位，收藏5个...")
        print(f"  含Agent岗位数: {len(agent_jobs)}")

        # 如果有Agent相关岗位，尝试收藏
        collected = 0
        for job in agent_jobs[:5]:
            try:
                print(f"  正在收藏: {job['title'][:30]}")
                # Boss直聘收藏按钮
                await browser.click(f".job-list-item:nth-child({job['index']}) .collect-btn")
                await browser.wait(0.5)
                collected += 1
                print(f"  ✓ 已收藏 ({collected}/5)")
            except Exception as e:
                print(f"  收藏失败: {e}")

        print(f"  总计收藏: {collected} 个岗位")

        # ========================================
        # 步骤6: 总结核心技术需求
        # ========================================
        print("\n[步骤6] 总结核心技术需求...")

        # 从岗位详情提取关键词
        tech_keywords = [
            "Python", "LangChain", "RAG", "Agent", "LLM",
            "向量数据库", "ChromaDB", "Pinecone", "Milvus",
            "OpenAI", "Claude", "GPT", "Prompt Engineering",
            "Fine-tuning", "微调", "Embedding", "向量检索",
            "FastAPI", "Flask", "Django", "异步编程",
            "Redis", "MongoDB", "PostgreSQL",
            "Kubernetes", "Docker", "CI/CD",
            "TensorFlow", "PyTorch", "Transformers",
            "多模态", "语音识别", "OCR", "NLP",
        ]

        # 统计技术出现频率
        for job in job_data:
            detail_lower = job["detail"].lower()
            for tech in tech_keywords:
                if tech.lower() in detail_lower:
                    skill_summary[tech] = skill_summary.get(tech, 0) + 1

        # 排序并展示
        print("\n  核心技术需求排名:")
        sorted_skills = sorted(skill_summary.items(), key=lambda x: x[1], reverse=True)
        for tech, count in sorted_skills[:10]:
            print(f"    - {tech}: 出现 {count} 次")

        # ========================================
        # 步骤7: 对比自身技能，列出短板
        # ========================================
        print("\n[步骤7] 对比自身技能，列出短板...")

        # 假设的自身技能（可根据实际情况调整）
        my_skills = [
            "Python", "LangChain", "RAG", "Agent",
            "OpenAI", "Claude", "Prompt Engineering",
            "FastAPI", "异步编程", "Playwright",
            "Docker", "Git", "Linux",
        ]

        # 找出短板
        gaps = []
        for tech, count in sorted_skills:
            if tech not in my_skills and count >= 2:
                gaps.append((tech, count))

        print("\n  技能短板（市场需求>=2但未掌握）:")
        for tech, count in gaps:
            print(f"    - {tech}: 市场需求 {count} 次")

        # ========================================
        # 步骤8: 生成技能提升清单
        # ========================================
        print("\n[步骤8] 生成技能提升清单...")

        print("\n" + "=" * 60)
        print("【技能提升清单】")
        print("=" * 60)

        # 优先级分类
        high_priority = [g for g in gaps if g[1] >= 3]
        medium_priority = [g for g in gaps if g[1] == 2]

        print("\n【高优先级】（市场需求>=3次）:")
        for tech, count in high_priority:
            print(f"  ▶ {tech}")
            # 添加学习建议
            suggestions = {
                "向量数据库": "学习 ChromaDB/Milvus 使用，掌握向量检索原理",
                "Embedding": "理解 Embedding 模型原理，实践文本向量化",
                "微调": "学习 LLM Fine-tuning 流程，实践 LoRA 微调",
                "PyTorch": "掌握深度学习框架，理解神经网络基础",
                "Kubernetes": "学习容器编排，掌握 K8s 部署",
                "Redis": "学习缓存架构，掌握高性能数据存储",
            }
            if tech in suggestions:
                print(f"    建议: {suggestions[tech]}")

        print("\n【中优先级】（市场需求2次）:")
        for tech, count in medium_priority:
            print(f"  ▷ {tech}")

        print("\n【已掌握技能】:")
        for skill in my_skills:
            if skill_summary.get(skill, 0) >= 1:
                print(f"  ✓ {skill} (市场需求 {skill_summary[skill]} 次)")

        # 最终截图
        await browser.screenshot("final_result")

        # ========================================
        # 测试总结
        # ========================================
        print("\n" + "=" * 60)
        print("测试完成总结")
        print("=" * 60)
        print(f"  查看岗位: {len(job_data)} 个")
        print(f"  Agent相关岗位: {len(agent_jobs)} 个")
        print(f"  收藏岗位: {collected} 个")
        print(f"  技术技能统计: {len(skill_summary)} 种")
        print(f"  技能短板: {len(gaps)} 项")
        print("=" * 60)

        return {
            "jobs": job_data,
            "agent_jobs": agent_jobs,
            "skill_summary": skill_summary,
            "gaps": gaps,
            "collected": collected,
        }

    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"\n测试失败: {e}")

    finally:
        await browser.close()
        print("\n浏览器已关闭")


if __name__ == "__main__":
    print("正在运行Boss直聘求职搜索测试...")
    asyncio.run(boss_zhipin_test())