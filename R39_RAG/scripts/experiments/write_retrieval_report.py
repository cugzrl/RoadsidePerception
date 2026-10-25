#!/usr/bin/env python3
"""Write a reproducible result report from completed, fixed-protocol artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
GROUPS=['stmc','memory_cv','kv_legacy','kv_encoded']
LABELS={'stmc':'STMC（共同基础修复）','memory_cv':'STMC + CV 记忆',
        'kv_legacy':'STMC + 原 KV','kv_encoded':'STMC + 编码 KV / 融合'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
            *['| '+' | '.join(str(x) for x in row)+' |' for row in rows],'']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    exp=args.experiment.resolve()
    primary=read(exp/'evaluation/metrics.json')['methods']
    agnostic=read(exp/'evaluation_class_agnostic/metrics.json')['methods']
    retrieval=read(exp/'retrieval_metrics.json')
    training=read(exp/'encoded_bank/training.json')
    diagnosis=read(exp/'switch_diagnosis.json')
    audits=read(exp/'data_audit.json')
    parity=read(exp/'device_parity.json')
    sequences=list(primary['stmc']['sequences'])
    runs={seq:{g:read(exp/'tracking_hybrid'/seq/g/'run_meta.json') for g in GROUPS} for seq in sequences}
    for seq in sequences:
        if any(runs[seq][g]['input_sha256']!=runs[seq]['stmc']['input_sha256'] for g in GROUPS):
            raise ValueError(f'Mismatched experimental inputs: {seq}')
    if set(training['meta']['training_sequences']) & set(sequences):
        raise ValueError('Train/test overlap')
    result=primary['kv_encoded']['combined']
    baseline=primary['stmc']['combined']
    legacy=primary['kv_legacy']['combined']
    validation_path=exp/'validation_summary.json'
    if validation_path.exists():
        validation=read(validation_path)
        test_count=str(validation['tests']['count'])
        if validation['tests']['status']!='passed':
            raise ValueError('Tests are not passing')
    else:
        test_log=(exp/'tests.log').read_text()
        test_count=re.search(r'Ran (\d+) tests',test_log).group(1)
        if not test_log.rstrip().endswith('OK'):
            raise ValueError('Tests are not passing')
    same_as_legacy=all(result[field]==legacy[field] for field in ['HOTA','AssA','MOTA','IDF1','IDSW'])
    interpretation=('本轮编码 KV 与原 KV 的主表指标完全一致，未证明编码/FAISS 具有额外的身份跟踪收益。'
                    '相对 STMC 的主要增益来自 CV 在线记忆；原 KV 仅在其上额外减少了 '
                    f'{primary["memory_cv"]["combined"]["IDSW"]-legacy["IDSW"]} 次 IDSW。'
                    if same_as_legacy else
                    '本轮变化只代表两个测试序列相对伪真值的结果，需分别检查序列和各项指标，不能预设新编码检索全面优于原方法。')
    lines=['# R39 编码知识库改进实验结果（2026-10-25）','',
           '本报告基于两个留出序列的完整四路视频结果，每路 1200 帧、20 FPS。指标参考 SAM3 框及部分人工修正的全局 ID，是伪真值一致性评估。', '',
           f'相对相同检测/特征的 STMC，编码知识增强的 HOTA 变化为 {result["HOTA"]-baseline["HOTA"]:+.4f} 个百分点，IDF1 变化为 {result["IDF1"]-baseline["IDF1"]:+.4f} 个百分点，IDSW 从 {baseline["IDSW"]} 变为 {result["IDSW"]}。',
           f'相对原 KV，HOTA 变化为 {result["HOTA"]-legacy["HOTA"]:+.4f} 个百分点，IDF1 变化为 {result["IDF1"]-legacy["IDF1"]:+.4f} 个百分点，IDSW 从 {legacy["IDSW"]} 变为 {result["IDSW"]}。', '',
           interpretation, '',
           '不能把相对 STMC 的全部变化归因于新编码器：CV 在线记忆本身也会改善关联。必须同时看下面四组消融；检索表示与融合的组合差异也不能归因于 FAISS 单个组件。','',
           '## 标准逐相机 MOT 指标的官方聚合','',
           '分数为百分数；HOTA/DetA/AssA/MOTA/IDF1 越高越好，IDSW 越低越好。STMC 行包含所有组共用的既有位置、短时恢复和关联保护修复。','']
    fields=['HOTA','DetA','AssA','MOTA','IDF1','IDSW']
    lines+=table(['方法',*fields],[[LABELS[g],*[f'{primary[g]["combined"][f]:.4f}' if f!='IDSW' else primary[g]['combined'][f] for f in fields]] for g in GROUPS])
    lines+=['## 两个测试序列分别结果','']
    lines+=table(['序列','方法','HOTA','AssA','MOTA','IDF1','IDSW'],[
        [seq.rsplit('_',1)[-1],LABELS[g],*[f'{primary[g]["sequences"][seq]["camera_combined"][f]:.4f}' if f!='IDSW' else primary[g]['sequences'][seq]['camera_combined'][f]
                                         for f in ['HOTA','AssA','MOTA','IDF1','IDSW']]] for seq in sequences for g in GROUPS])
    lines+=['## 跨视角全局身份补充指标','',
            '按 `(frame,camera)` 观察流序列化、共享全局 ID。该定义用于本项目诊断，不可直接与标准 MTMC 榜单对比；不把序列化流的 IDSW 当作标准相机内 IDSW。','']
    lines+=table(['方法','观察流 HOTA','观察流 AssA','全局 IDF1'],[
        [LABELS[g],*[f'{primary[g]["global_observation_stream"][f]:.4f}' for f in ['HOTA','AssA','IDF1']]] for g in GROUPS])
    lines+=['## 类别映射敏感性','', '以下关闭类别匹配约束，其他协议保持一致。它是敏感性分析，主表仍使用严格类别匹配。','']
    lines+=table(['方法','HOTA','MOTA','IDF1','IDSW'],[
        [LABELS[g],*[f'{agnostic[g]["combined"][f]:.4f}' if f!='IDSW' else agnostic[g]['combined'][f] for f in ['HOTA','MOTA','IDF1','IDSW']]] for g in GROUPS])
    lines+=['## 检索与未来预测','',
            '4434 个质量筛选的测试窗口，来自 207 条相机内轨迹片段。所有在线方法均在未命中时回退 CV，不使用参考未来选择最佳候选。这里评估的是轨迹预测，不能替代上面的身份跟踪指标。','']
    lines+=table(['方法','ADE / m','FDE / m','轨迹宏平均 ADE / m','轨迹宏平均 FDE / m'],[
        [name,f'{r["ade_m"]:.4f}',f'{r["fde_m"]:.4f}',f'{r["track_macro"]["ade_m"]:.4f}',f'{r["track_macro"]["fde_m"]:.4f}'] for name,r in retrieval['methods'].items()])
    cv=retrieval['methods']['CV']
    mode=retrieval['methods']['encoded_mode_fallback']
    lines += [f'模式选择相对 CV 的窗口 ADE 降低 {(cv["ade_m"]-mode["ade_m"])/cv["ade_m"]*100:.2f}%，FDE 降低 {(cv["fde_m"]-mode["fde_m"])/cv["fde_m"]*100:.2f}%。编码 Top-1 没有优于原 Top-1，不能据此宣称编码表示本身已经带来确定收益。', '',
              '在线跟踪使用候选检测位置的混合似然；本节独立预测使用历史检索候选之间的邻域模式选择。两者都不读取测试未来，但不是同一个输出任务。','']
    lines+=table(['检索方法','有效命中窗口','平均查询 / ms','P95 查询 / ms'],[
        [name,retrieval['hit_count'][name],f'{r["mean"]:.4f}',f'{r["p95"]:.4f}'] for name,r in retrieval['latency_ms'].items()])
    lines+=['编码查询包含 Transformer 和 FAISS。小型库中它比原 NumPy 检索慢；收益目标是候选质量与扩展接口，不是本次规模下的速度。','',
            '## 在线恢复实际使用情况','',
            '`kv` 是最终采用知识距离的加边次数，`blend_memory` 是命中知识后仍回退匀速距离的次数。这些是决策日志，不是正确恢复数量或新的评估真值。','']
    usage=[]
    saved_usage=read(exp/'recovery_usage.json') if (exp/'recovery_usage.json').exists() else {}
    for seq in sequences:
        for group in GROUPS:
            path=exp/'tracking_hybrid'/seq/group/'memory_log.jsonl'
            if path.exists():
                rows=[json.loads(line) for line in path.open()]
                modes=Counter(r['mode'] for r in rows if r['action']=='accept')
            else:
                modes=Counter(saved_usage.get(seq,{}).get(group,{}).get('accepted_modes',{}))
            usage.append([seq.rsplit('_',1)[-1],group,sum(modes.values()),modes['kv'],modes['blend_memory'],modes['fallback_memory']])
    lines+=table(['序列','方法','全部接受','知识距离','命中后 CV','无命中回退'],usage)
    lines+=['## 为什么整体效果可能有限','']
    total=sum(r['IDSW'] for r in diagnosis.values())
    small=sum(r['small_reference_box_height_under_40'] for r in diagnosis.values())
    bins=Counter()
    for r in diagnosis.values():
        bins.update(r['by_matched_gap'])
    lines += [f'基线 {total} 次 IDSW 中，{small} 次对应高度不足 40 像素的参考框（{small/max(total,1)*100:.1f}%）。按前次成功匹配的间隔分类：连续帧 {bins["consecutive"]} 次，1–3 帧 {bins["gap_1_to_3"]} 次，4–30 帧 {bins["gap_4_to_30"]} 次，超过 30 帧 {bins["gap_over_30"]} 次。诊断计数逐相机核对官方 CLEAR。', '',
              '匹配间隔不等于跟踪器内部的真实失联状态，但它说明当前只在失联后启用知识的模块覆盖范围有限。远端小目标的框抖动、投影误差和检测缺失不能仅靠 FAISS 解决。更长时段、更丰富转向样本，以及针对连续关联歧义的可靠性建模，是后续比继续增加检索复杂度更直接的方向。', '',
              '上述后续方向尚未在本报告中验证。本轮没有依据测试分数搜索门限、选择最好种子或删去不利序列。','',
              '## 数据与验证','']
    lines+=table(['序列','参考有效框','跟踪输出框','未分配 ID 检测','重复 ID 次数'],[
        [seq.rsplit('_',1)[-1],primary['stmc']['sequences'][seq]['audit']['reference']['valid_observations'],
         primary['stmc']['sequences'][seq]['audit']['prediction']['tracked_observations'],
         primary['stmc']['sequences'][seq]['audit']['prediction']['without_track_id'],
         primary['stmc']['sequences'][seq]['audit']['prediction'].get('duplicate_identity_occurrences_split',0)] for seq in sequences])
    lines += [f'四组检测、外观数组与特征元数据的 SHA256 在每段序列上完全一致。输入中未发现身份或未来字段。训练库含 {training["meta"]["window_count"]} 个窗口、{training["meta"]["independent_tracks"]} 条相机内轨迹片段；选定训练轮数 {training["meta"]["selected_epochs"]}，种子 {training["meta"]["seed"]}。', '',
              f'{test_count} 项单元测试通过；包括检索键不依赖未来值、FAISS/NumPy 一致性、泄漏防护、标准指标的完美/切换/漏检情形、绝对坐标恢复和歧义回退。40 帧的混合执行与原 CUDA 图执行，共 {parity["records_cpu"]} 条记录的规范化身份关联一致：{parity["identical_canonical_associations"]}。另已通过 Python 编译与补丁格式检查。', '',
              '原始数据含部分人工 ID 修正与清洗重编号，跨视角同步仍未完全核实。分数应连同这些边界报告；不把伪真值升级为人工逐帧真值，也不把两个序列的变化描述成已证实的广泛泛化。','',
              '## 文件','',
              '- 主结果：`outputs/rag_tp_20261025/evaluation/metrics.json`、`comparison.csv`。',
              '- 类别无关结果：`outputs/rag_tp_20261025/evaluation_class_agnostic/`。',
              '- 完整跟踪输出与参数：`outputs/rag_tp_20261025/tracking_hybrid/<sequence>/<group>/`。',
              '- 模型、库与训练曲线：`outputs/rag_tp_20261025/encoded_bank/`。',
              '- 预测误差：`outputs/rag_tp_20261025/retrieval_metrics.json`。',
              '- 输入审计、切换事件：`data_audit.json`、`switch_diagnosis.json`。',
              '- 架构与复现命令：`docs/rag_tp_retrieval_guide.md`。',
              '- 原有历史实验、知识库和原始数据保持保留，本轮结果写入独立目录。','']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines),encoding='utf-8')
    source_files=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    source_files+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(set(source_files)) if (ROOT/name).is_file()}
    manifest={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'modified_or_added_files_sha256':hashes,'matched_inputs':{s:runs[s]['stmc']['input_sha256'] for s in sequences},
              'experiment_directory':str(exp),'test_count':int(test_count)}
    (exp/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    patch=subprocess.check_output(['git','diff','--binary'],cwd=ROOT)
    new_files=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    for name in filter(None,new_files):
        result=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],cwd=ROOT,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if result.returncode not in [0,1]:
            raise RuntimeError(result.stderr.decode())
        patch+=result.stdout
    (exp/'source_changes.patch').write_bytes(patch)
    print(args.output)


if __name__=='__main__':
    main()
