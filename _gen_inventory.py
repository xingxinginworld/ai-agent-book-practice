# -*- coding: utf-8 -*-
import os, re

SRC = r'D:\Project\FDE转型之路\FDE自验证\AI-Agent知识库\_work'
OUT = r'D:\Project\FDE转型之路\FDE自验证\AI-Agent实验跟练'
os.makedirs(os.path.join(OUT, 'experiments'), exist_ok=True)

files = [f for f in sorted(os.listdir(SRC)) if f.endswith('.md')]

def ch_of(f):
    m = re.search(r'_(\d+)_', f)
    return int(m.group(1)) if m else 0

rows = []
pat = re.compile(r'^实验\s*(\d+)-(\d+)\s*([★]*)\s*[：:]\s*(.+?)\s*$')
for f in files:
    ch = ch_of(f)
    for ln in open(os.path.join(SRC, f), encoding='utf-8'):
        m = pat.match(ln.strip())
        if m:
            rows.append([int(m.group(1)), int(m.group(2)), len(m.group(3)), m.group(4)])
rows.sort(key=lambda r: (r[0], r[1]))

# ---------- 分类规则 ----------
def sec_type(title, ch):
    t = title
    if any(k in t for k in ['消融', '对比', '比较', '对照']):
        return '对照·消融·对比'
    if any(k in t for k in ['评估', '基准', '排行榜', '评测', '归因', '测量']):
        return '评测构建'
    if any(k in t for k in ['训练', 'SFT', 'RL', '预训练', '蒸馏', '后训练', '微调']):
        return '模型训练/后训练'
    if any(k in t for k in ['部署', '构建', 'MCP', '服务', '系统', 'Skill', '感知',
                            '执行', '协作', '诊断', '解析', '生成', '开发', '自动化',
                            '创造', '翻译', '电话', '搜集', '编排']):
        return '工程搭建'
    if any(k in t for k in ['可视化', '观测', '运行', '演示', '漫游', '整理']):
        return '直觉建立/观测演示'
    fb = {1: '直觉建立/观测演示', 2: '上下文工程实践', 3: '记忆/RAG实践',
          4: '工具/MCP搭建', 5: 'Coding/通用Agent', 6: '交互/多模态实践',
          7: '评测构建', 8: '模型训练/后训练', 9: '持续进化/自改进', 10: '多Agent协作'}
    return fb.get(ch, '工程搭建')

GPU = {'8-3','8-4','8-5','8-6','8-7','8-8','8-9','8-10','8-11','8-12',
       '8-13','8-14','8-15','8-16','8-17','8-18','8-19'}
HARD = {'6-10','6-12','6-13','6-14'}
API = {'1-2','1-3','5-6','5-7','5-8','5-12','5-13','5-15','6-7',
       '7-13','10-3','10-4','10-6'}

def resource(ch, num):
    key = f'{ch}-{num}'
    if key in GPU:  return '需GPU训练'
    if key in HARD: return '专用硬件'
    if key in API:  return '需API密钥'
    return '轻量本地'

def tier(res):
    return {'轻量本地': '必做', '需API密钥': '选做', '需GPU训练': '观察',
            '专用硬件': '观察'}[res]

ROLE = {1:'通用型',2:'通用型',3:'工程向',4:'工程向',5:'工程向',6:'工程向',
        7:'工程向(评测)',8:'算法/训练向',9:'工程向',10:'工程向'}

FIRST = (2, 1)  # 首个跟练单元

records = []
for ch, num, stars, title in rows:
    res = resource(ch, num)
    rec = {
        'no': f'{ch}-{num}', 'ch': ch, 'num': num, 'stars': '★' * stars,
        'title': title, 'sec': sec_type(title, ch), 'res': res,
        'tier': tier(res), 'role': ROLE[ch],
        'status': '进行中(首单元)' if (ch, num) == FIRST else '待跟练',
    }
    records.append(rec)

# ---------- 总表 md ----------
ch_names = {1:'AI Agent 入门',2:'上下文工程',3:'用户记忆和知识库',4:'工具',
            5:'Coding Agent 与通用 Agent',6:'交互：观察与动作空间的扩展',
            7:'Agent 的评估',8:'模型后训练',9:'Agent 的持续进化',10:'多 Agent 协作'}

def gh(ch, num):
    return f'experiments/ch{ch:02d}/exp{ch}-{num}/README.md'

lines = []
lines.append('# 《深入理解 AI Agent》实验盘点与跟练计划总表\n')
lines.append('> 来源：全书正文（第1–10章 + 后记）"实验 X-Y" 章节标题系统性提取。')
lines.append('> 共 **%d** 个实验。三档说明：**必做**=本地轻量可实跑；**选做**=需外部 API/云密钥；**观察**=需 GPU 训练或专用硬件，先理解不实跑。\n' % len(records))

# 统计
from collections import Counter
tier_c = Counter(r['tier'] for r in records)
res_c = Counter(r['res'] for r in records)
lines.append('## 一、总体统计\n')
lines.append('- 实验总数：**%d**' % len(records))
lines.append('- 三档分布：必做 %d · 选做 %d · 观察 %d' % (tier_c['必做'], tier_c['选做'], tier_c['观察']))
lines.append('- 资源分布：轻量本地 %d · 需API密钥 %d · 需GPU训练 %d · 专用硬件 %d' %
             (res_c.get('轻量本地',0), res_c.get('需API密钥',0), res_c.get('需GPU训练',0), res_c.get('专用硬件',0)))
lines.append('- 难度分布：★ %d · ★★ %d · ★★★ %d' %
             (sum(1 for r in records if r['stars']=='★'),
              sum(1 for r in records if r['stars']=='★★'),
              sum(1 for r in records if r['stars']=='★★★')))
lines.append('')

# 分章统计
lines.append('## 二、分章概览\n')
lines.append('| 章 | 主题 | 实验数 | 必做 | 选做 | 观察 |')
lines.append('| :--: | --- | :--: | :--: | :--: | :--: |')
for ch in range(1, 11):
    cr = [r for r in records if r['ch'] == ch]
    if not cr: continue
    b = sum(1 for r in cr if r['tier']=='必做')
    s = sum(1 for r in cr if r['tier']=='选做')
    o = sum(1 for r in cr if r['tier']=='观察')
    lines.append('| 第%d章 | %s | %d | %d | %d | %d |' % (ch, ch_names[ch], len(cr), b, s, o))
lines.append('')

# 完整明细
lines.append('## 三、完整明细（%d 个）\n' % len(records))
lines.append('| # | 编号 | 章节 | 难度 | 二级分类 | 资源档 | 三档 | 适合对象 | 标题 | 状态 |')
lines.append('| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | --- | :--: |')
for i, r in enumerate(records, 1):
    chp = '第%d章' % r['ch']
    lines.append('| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |' %
                 (i, r['no'], chp, r['stars'], r['sec'], r['res'], r['tier'], r['role'], r['title'], r['status']))
lines.append('')

open(os.path.join(OUT, '实验盘点总表.md'), 'w', encoding='utf-8').write('\n'.join(lines))
print('已写出 实验盘点总表.md，共 %d 行，实验 %d 个' % (len(lines), len(records)))

# 保存 records 供 README 索引与后续脚本复用
import json
json.dump(records, open(os.path.join(OUT, '_records.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('已写出 _records.json')
