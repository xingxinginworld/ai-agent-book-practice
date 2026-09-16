# -*- coding: utf-8 -*-
import json, os
from collections import Counter

OUT = r'D:\Project\FDE转型之路\FDE自验证\AI-Agent实验跟练'
recs = json.load(open(os.path.join(OUT, '_records.json'), encoding='utf-8'))

ch_names = {1:'AI Agent 入门',2:'上下文工程',3:'用户记忆和知识库',4:'工具',
            5:'Coding Agent 与通用 Agent',6:'交互：观察与动作空间的扩展',
            7:'Agent 的评估',8:'模型后训练',9:'Agent 的持续进化',10:'多 Agent 协作'}

tier_c = Counter(r['tier'] for r in recs)
res_c = Counter(r['res'] for r in recs)
star_c = Counter(r['stars'] for r in recs)

L = []
L.append('# ai-agent-book-practice')
L.append('')
L.append('《深入理解 AI Agent》全书 **109** 个实验的系统性跟练仓库。')
L.append('')
L.append('> 原书配套代码见作者仓库 [bojieli/ai-agent-book](https://github.com/bojieli/ai-agent-book)（已开源）。')
L.append('> 本仓库在其基础上，按"实验 X-Y"逐实验产出**精简可运行 + 中文逐行注释**版本，并记录跟练过程，用于学习沉淀与公众号连载。')
L.append('')
L.append('## 跟练约定')
L.append('- 每个实验一个目录：`experiments/chXX/expX-Y/README.md`，内含：目标 → 前置 → 环境 → 分步 → 核心代码+注释 → 备注/踩坑 → 公众号记录要点。')
L.append('- **三档**：🔵 必做（本地轻量可实跑）· 🟡 选做（需外部 API/云密钥）· ⚪ 观察（需 GPU 训练或专用硬件，先理解不实跑）。')
L.append('- 公众号：每完成一个实验对应一篇图文记录。')
L.append('')
L.append('## 总进度（自动生成）')
L.append('- 实验总数：**%d**' % len(recs))
L.append('- 三档：🔵 必做 %d · 🟡 选做 %d · ⚪ 观察 %d' % (tier_c['必做'], tier_c['选做'], tier_c['观察']))
L.append('- 资源：轻量本地 %d · 需API密钥 %d · 需GPU训练 %d · 专用硬件 %d' %
         (res_c.get('轻量本地',0), res_c.get('需API密钥',0), res_c.get('需GPU训练',0), res_c.get('专用硬件',0)))
L.append('- 难度：★ %d · ★★ %d · ★★★ %d' % (star_c.get('★',0), star_c.get('★★',0), star_c.get('★★★',0)))
L.append('- 当前进度：实验 2-1（进行中，首单元）')
L.append('')
L.append('## 实验索引')
L.append('')
for ch in range(1, 11):
    cr = [r for r in recs if r['ch'] == ch]
    if not cr: continue
    L.append('### 第%d章 %s（%d 个）' % (ch, ch_names[ch], len(cr)))
    L.append('')
    L.append('| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |')
    L.append('| :--: | :--: | :--: | :--: | :--: | --- | :--: |')
    for r in cr:
        path = 'experiments/ch%02d/exp%s/README.md' % (r['ch'], r['no'])
        link = '[%s](%s)' % (r['no'], path)
        L.append('| %s | %s | %s | %s | %s | %s | %s |' %
                 (link, r['stars'], r['sec'], r['res'], r['tier'], r['title'], r['status']))
    L.append('')

L.append('## 参考')
L.append('- 原书：《深入理解 AI Agent》（博杰力 著）')
L.append('- 官方代码仓库：https://github.com/bojieli/ai-agent-book')
L.append('')

open(os.path.join(OUT, 'README.md'), 'w', encoding='utf-8').write('\n'.join(L))
print('已写出 README.md，共 %d 行' % len(L))
