"""Build the six ordered Colab notebooks for the frozen Stage C v3 study."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1] / "notebooks" / "titans_stage_c"
PIN = "df20b8614ca3165d365dd945948ca24d9495d27e"


def cell(source: str, kind: str = "code") -> dict[str, object]:
    payload: dict[str, object] = {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        payload.update(execution_count=None, outputs=[])
    return payload


def write(name: str, title: str, cells: list[str], *, gpu: str | None = None) -> None:
    metadata: dict[str, object] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    if gpu:
        metadata.update(accelerator="GPU", colab={"gpuType": gpu})
    payload = {
        "cells": [cell(f"# {title}\n", "markdown"), *[cell(item) for item in cells]],
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / name).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


CONFIG = f"""# USER CONFIGURATION
REPO_URL='https://github.com/Gonza10V/SeqTrainer.git'
GIT_REF='{PIN}'
DRIVE_ROOT='/content/drive/MyDrive/SeqTrainerStageC'
DATASET_NAME='nonoverlap_6mer_v1'
TAXONOMY_MANIFEST=f'{{DRIVE_ROOT}}/stage_c_dataset/manifests/accession_manifest.parquet'
ACCESSION_MANIFEST=TAXONOMY_MANIFEST
ANI_MEMBERSHIP=f'{{DRIVE_ROOT}}/stage_c_dataset/manifests/ani99_membership.parquet'
ANI_PAIRS=f'{{DRIVE_ROOT}}/stage_c_dataset/manifests/skani_triangle.tsv'
"""

BOOTSTRAP = """from pathlib import Path
from google.colab import drive
import json, shutil, subprocess, sys
mount=Path('/content/drive')
if not (mount/'MyDrive').is_dir(): drive.mount(str(mount),timeout_ms=120000)
repo=Path('/content/SeqTrainer')
if not repo.exists(): subprocess.run(['git','clone',REPO_URL,str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','origin'],check=True)
subprocess.run(['git','-C',str(repo),'checkout',GIT_REF],check=True)
subprocess.run([sys.executable,'-m','pip','install','-e',f'{repo}[torch,bacteria-titan]'],check=True)
dataset=Path(DRIVE_ROOT)/'stage_c_dataset/ordered_streams'/DATASET_NAME
panels=Path(DRIVE_ROOT)/'study/stage_c_ecoli_medium_deep_memory_v3/panels'
PROTOCOL=repo/'studies/stage_c_ecoli_medium_deep_memory_v3/protocol.json'
STUDY_ROOT=Path(DRIVE_ROOT)/'study/stage_c_ecoli_medium_deep_memory_v3'
subprocess.run(['seqtrainer-titans-stage-c-study','initialize','--protocol',str(PROTOCOL),'--study-root',str(STUDY_ROOT)],check=True)
def run_logged(root,label,command):
    root.mkdir(parents=True,exist_ok=True)
    try:
        subprocess.run(['seqtrainer-titans-stage-c-colab-run','--run-dir',str(root),'--label',label,'--repo',str(repo),'--',*command],check=True)
    except subprocess.CalledProcessError:
        for path in (root/'FAILED.txt',root/'logs'/f'{label}.log'):
            if path.exists(): print(path.read_text(errors='replace')[-20000:])
        raise
def record_once(run_id,artifact,tier):
    marker=STUDY_ROOT/'record_markers'/f'{run_id}.json'
    if marker.exists():
        print('Ledger record already exists:',marker); return
    subprocess.run(['seqtrainer-titans-stage-c-study','record','--protocol',str(PROTOCOL),'--study-root',str(STUDY_ROOT),'--run-id',run_id,'--evidence-tier',tier,'--artifact',str(artifact)],check=True)
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps({'run_id':run_id,'artifact':str(artifact)},indent=2)+'\\n')
"""

DEEP_FLAGS = """deep_flags=['--memory-architecture','paper_residual_mlp_v2','--memory-depth','2','--memory-expansion-factor','4','--memory-projection-convolution-kernel','4','--memory-normalize-queries-and-keys','--memory-gate-granularity','per_layer_channel','--memory-recurrence-policy','paper_exact','--memory-surprise-clip-norm','none','--memory-alpha-initial','0.001','--memory-eta-initial','0.9','--memory-theta-initial','0.001','--memory-associative-loss-reduction','sum','--memory-max-gradient-rms','none','--memory-max-gradient-rms-ratio','none','--memory-theta-max','1.0']
"""

write(
    "03j_stage_c_v3_freeze_panels_and_c16_baseline.ipynb",
    "Stage C 03j — freeze nested complete-replicon panels and c16 baseline",
    [
        CONFIG,
        BOOTSTRAP,
        """for path in map(Path,(ACCESSION_MANIFEST,ANI_MEMBERSHIP,ANI_PAIRS)):
    if not path.is_file(): raise FileNotFoundError(path)
if not (panels/'panel_summary.json').is_file():
    run_logged(panels,'freeze_ecoli_panels',['seqtrainer-titans-stage-c-panel','freeze','--dataset-dir',str(dataset),'--accession-manifest',ACCESSION_MANIFEST,'--ani-membership',ANI_MEMBERSHIP,'--ani-pairs',ANI_PAIRS,'--output-dir',str(panels)])
for name in ('e25','e100','e250','e100_additions','validation','test'):
    subprocess.run(['seqtrainer-titans-stage-c-panel','validate','--dataset-dir',str(dataset),'--panel-manifest',str(panels/f'{name}.json')],check=True)
record_once('ecoli_panel_freeze_v1',panels,'engineering')
print((panels/'panel_summary.json').read_text())
""",
        """import torch
if not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime for the c16 baseline.')
baseline=Path(DRIVE_ROOT)/'runs/c17_v3_c16_broad_baseline'
c16=Path(DRIVE_ROOT)/'runs/c16_deep_adaptive_5m_paper_exact/latest.pt'
run_logged(baseline,'evaluate_c16',['seqtrainer-titans-stage-c-evaluate','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'validation.json'),'--run',f'c16={c16}','--output-dir',str(baseline/'evaluation'),'--split','val','--comparison-mode','partial','--device','cuda','--protocol',str(PROTOCOL),'--run-id','c16_broad_ecoli_baseline_v1'])
if not shutil.which('prodigal'):
    subprocess.run(['apt-get','update'],check=True); subprocess.run(['apt-get','install','-y','prodigal'],check=True)
run_logged(baseline,'generate_c16',['seqtrainer-titans-stage-c-generate','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'validation.json'),'--taxonomy-manifest',TAXONOMY_MANIFEST,'--checkpoint',str(c16),'--output-dir',str(baseline/'generation_t0p6'),'--split','val','--species','Escherichia coli','--prompts','4','--prompt-tokens','128','--new-tokens','1024','--temperatures','0.6','--top-k','1024','--top-p','0.99','--device','cuda','--memory-mode','adaptive','--prodigal',shutil.which('prodigal'),'--protocol',str(PROTOCOL),'--run-id','c16_broad_ecoli_baseline_v1'])
record_once('c16_broad_ecoli_baseline_v1',baseline,'exploratory')
print('SHARE THIS DIRECTORY:',baseline)
""",
    ],
    gpu="T4",
)

write(
    "03k_stage_c_v3_medium_a100_qualification.ipynb",
    "Stage C 03k — Medium A100 capacity and numeric qualification",
    [
        CONFIG,
        BOOTSTRAP,
        """import torch
if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0).upper(): raise RuntimeError('Select an A100 runtime.')
root=Path(DRIVE_ROOT)/'runs/c18_v3_medium_a100_qualification'
command=['seqtrainer-titans-stage-c-capacity','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'e25.json'),'--validation-panel-manifest',str(panels/'validation.json'),'--output-dir',str(root),'--require','A100','--horizons','3','--variants','exact_sdpa_fp32','exact_sdpa_bfloat16','--steps','10','--batch-size','1','--block-count','12','--d-model','256','--num-heads','8','--persistent-tokens','4','--memory-depth','2','--memory-architecture','paper_residual_mlp_v2','--memory-recurrence-policy','paper_exact','--validation-segments','32']
run_logged(root,'medium_capacity',command)
report=json.loads((root/'capacity_matrix.json').read_text())
fp32=next(x for x in report['results'] if x['variant']=='exact_sdpa_fp32')
total=torch.cuda.get_device_properties(0).total_memory
eligible=[x for x in report['results'] if x['available'] and x['peak_allocated_bytes']<=0.70*total and abs(x['validation_bpb']-fp32['validation_bpb'])<=0.005 and x['finite']]
if not eligible: raise RuntimeError('No qualified activation; do not run 03l.')
best=max(eligible,key=lambda x:x['bases_per_second'])
selection={'format_version':1,'passed':True,'activation':{'exact_sdpa_fp32':'float32','exact_sdpa_bfloat16':'bfloat16'}[best['variant']],'batch_size':1,'selected':best,'total_gpu_bytes':total}
(root/'qualification_selection.json').write_text(json.dumps(selection,indent=2,sort_keys=True)+'\\n')
record_once('medium_a100_qualification_v1',root,'engineering')
print(json.dumps(selection,indent=2)); print('SHARE THIS DIRECTORY:',root)
""",
    ],
    gpu="A100",
)

TRAIN_COMMON = CONFIG + "RUN_NAME='c19_v3_medium_adaptive_e25'\nRUN_ID='medium_adaptive_e25_v1'\nPANEL='e25.json'\n"
write(
    "03l_stage_c_v3_medium_adaptive_e25.ipynb",
    "Stage C 03l — Medium adaptive E25 complete-panel training",
    [
        TRAIN_COMMON,
        BOOTSTRAP,
        DEEP_FLAGS + """import torch
if 'A100' not in torch.cuda.get_device_name(0).upper(): raise RuntimeError('Select an A100 runtime.')
qualification=Path(DRIVE_ROOT)/'runs/c18_v3_medium_a100_qualification/qualification_selection.json'
q=json.loads(qualification.read_text())
if not q['passed']: raise RuntimeError('03k did not qualify this run.')
root=Path(DRIVE_ROOT)/'runs'/RUN_NAME
command=['seqtrainer-titans-stage-c-train','--dataset-dir',str(dataset),'--panel-manifest',str(panels/PANEL),'--validation-panel-manifest',str(panels/'validation.json'),'--run-dir',str(root),'--memory-mode','adaptive','--horizon','3','--batch-size',str(q['batch_size']),'--require-panel-completion','--scheduler-policy','stateful_rotation','--scheduler-burst-segments','96','--checkpoint-every','250','--learning-rate','3e-5','--min-learning-rate','3e-6','--lr-warmup-bases','2000000','--lr-decay-bases','100000000','--weight-decay','0.1','--gradient-clip-norm','0.5','--activation',q['activation'],'--block-count','12','--d-model','256','--num-heads','8','--persistent-tokens','4',*deep_flags,'--protocol',str(PROTOCOL),'--run-id',RUN_ID]
run_logged(root,'train_e25',command)
run_logged(root,'architecture_e25',['seqtrainer-titans-stage-c-architecture','--checkpoint',str(root/'latest.pt'),'--output-dir',str(root)])
run_logged(root,'resume_verify_e25',['seqtrainer-titans-stage-c-resume-verify','--dataset-dir',str(dataset),'--panel-manifest',str(panels/PANEL),'--checkpoint',str(root/'latest.pt'),'--output',str(root/'resume_verification.json'),'--device','cuda'])
record_once(RUN_ID,root,'exploratory')
print((root/'MODEL_ARCHITECTURE.txt').read_text()); print('SHARE THIS DIRECTORY:',root)
""",
    ],
    gpu="A100",
)

ANALYSIS_CONFIG = CONFIG + "RUN_NAME='c19_v3_medium_adaptive_e25'\nRUN_ID='medium_adaptive_e25_analysis_v1'\nBASELINE_RUN='c17_v3_c16_broad_baseline'\nSTAGE='e25'\n"
write(
    "03m_stage_c_v3_medium_e25_analysis_and_gate.ipynb",
    "Stage C 03m — E25 held-out, memory, generation, and frozen scale gate",
    [
        ANALYSIS_CONFIG,
        BOOTSTRAP,
        """import torch
if not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime.')
root=Path(DRIVE_ROOT)/'runs'/RUN_NAME; checkpoint=root/'latest.pt'; out=root/'scale_analysis_v1'
run_logged(out,'evaluate_e25',['seqtrainer-titans-stage-c-evaluate','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'validation.json'),'--run',f'adaptive={checkpoint}','--output-dir',str(out/'evaluation'),'--split','val','--comparison-mode','partial','--device','cuda','--protocol',str(PROTOCOL),'--run-id',RUN_ID])
run_logged(out,'memory_behavior_e25',['seqtrainer-titans-stage-c-memory-behavior','--checkpoint',str(checkpoint),'--output',str(out/'memory_behavior.json'),'--pairs','64','--device','cuda','--protocol',str(PROTOCOL),'--run-id',RUN_ID])
run_logged(out,'memory_trace_e25',['seqtrainer-titans-stage-c-memory-trace','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'validation.json'),'--checkpoint',str(checkpoint),'--output-dir',str(out/'memory_trace'),'--split','val','--memory-mode','adaptive','--max-streams','12','--max-segments','512','--taxonomy-manifest',TAXONOMY_MANIFEST,'--taxonomy-rank','species','--device','cuda','--protocol',str(PROTOCOL),'--run-id',RUN_ID])
if not shutil.which('prodigal'):
    subprocess.run(['apt-get','update'],check=True); subprocess.run(['apt-get','install','-y','prodigal'],check=True)
run_logged(out,'generation_e25',['seqtrainer-titans-stage-c-generate','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'validation.json'),'--taxonomy-manifest',TAXONOMY_MANIFEST,'--checkpoint',str(checkpoint),'--output-dir',str(out/'generation_t0p6'),'--split','val','--species','Escherichia coli','--prompts','4','--prompt-tokens','128','--new-tokens','1024','--temperatures','0.6','--top-k','1024','--top-p','0.99','--device','cuda','--memory-mode','adaptive','--prodigal',shutil.which('prodigal'),'--protocol',str(PROTOCOL),'--run-id',RUN_ID])
baseline=Path(DRIVE_ROOT)/'runs'/BASELINE_RUN
subprocess.run(['seqtrainer-titans-stage-c-scale-gate','--stage',STAGE,'--baseline-evaluation',str(baseline/'evaluation/evaluation.json'),'--baseline-name','c16','--candidate-evaluation',str(out/'evaluation/evaluation.json'),'--candidate-name','adaptive','--memory-behavior',str(out/'memory_behavior.json'),'--baseline-generation',str(baseline/'generation_t0p6/generation_evaluation.json'),'--candidate-generation',str(out/'generation_t0p6/generation_evaluation.json'),'--output-dir',str(out/'gate')],check=True)
record_once(RUN_ID,out,'exploratory')
print((out/'gate/SCALE_GATE.md').read_text()); print('SHARE THIS DIRECTORY:',out)
""",
    ],
    gpu="A100",
)

write(
    "03n_stage_c_v3_medium_adaptive_e100_increment.ipynb",
    "Stage C 03n — gated E100-minus-E25 warm-start continuation",
    [
        CONFIG + "RUN_NAME='c20_v3_medium_adaptive_e100_increment'\nRUN_ID='medium_adaptive_e100_increment_v1'\n",
        BOOTSTRAP,
        DEEP_FLAGS + """import torch
gate=Path(DRIVE_ROOT)/'runs/c19_v3_medium_adaptive_e25/scale_analysis_v1/gate/scale_gate.json'
if not json.loads(gate.read_text())['proceed']: raise RuntimeError('E25 gate says STOP; 03n is intentionally blocked.')
q=json.loads((Path(DRIVE_ROOT)/'runs/c18_v3_medium_a100_qualification/qualification_selection.json').read_text())
parent=Path(DRIVE_ROOT)/'runs/c19_v3_medium_adaptive_e25/latest.pt'; root=Path(DRIVE_ROOT)/'runs'/RUN_NAME
command=['seqtrainer-titans-stage-c-train','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'e100_additions.json'),'--validation-panel-manifest',str(panels/'validation.json'),'--run-dir',str(root),'--warm-start-checkpoint',str(parent),'--no-resume','--memory-mode','adaptive','--horizon','3','--batch-size',str(q['batch_size']),'--require-panel-completion','--scheduler-policy','stateful_rotation','--scheduler-burst-segments','96','--checkpoint-every','250','--learning-rate','3e-5','--min-learning-rate','3e-6','--lr-warmup-bases','2000000','--lr-decay-bases','100000000','--weight-decay','0.1','--gradient-clip-norm','0.5','--activation',q['activation'],'--block-count','12','--d-model','256','--num-heads','8','--persistent-tokens','4',*deep_flags,'--protocol',str(PROTOCOL),'--run-id',RUN_ID]
run_logged(root,'train_e100_increment',command)
run_logged(root,'resume_verify_e100',['seqtrainer-titans-stage-c-resume-verify','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'e100_additions.json'),'--checkpoint',str(root/'latest.pt'),'--output',str(root/'resume_verification.json'),'--device','cuda'])
record_once(RUN_ID,root,'exploratory')
print('SHARE THIS DIRECTORY:',root)
""",
    ],
    gpu="A100",
)

write(
    "03o_stage_c_v3_medium_e100_analysis_and_gate.ipynb",
    "Stage C 03o — E100 analysis and matched-control allocation gate",
    [
        CONFIG + "RUN_NAME='c20_v3_medium_adaptive_e100_increment'\nRUN_ID='medium_adaptive_e100_analysis_v1'\nSTAGE='e100'\n",
        BOOTSTRAP,
        """import torch
root=Path(DRIVE_ROOT)/'runs'/RUN_NAME; checkpoint=root/'latest.pt'; out=root/'scale_analysis_v1'
run_logged(out,'evaluate_e100',['seqtrainer-titans-stage-c-evaluate','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'validation.json'),'--run',f'adaptive={checkpoint}','--output-dir',str(out/'evaluation'),'--split','val','--comparison-mode','partial','--device','cuda','--protocol',str(PROTOCOL),'--run-id',RUN_ID])
run_logged(out,'memory_behavior_e100',['seqtrainer-titans-stage-c-memory-behavior','--checkpoint',str(checkpoint),'--output',str(out/'memory_behavior.json'),'--pairs','64','--device','cuda','--protocol',str(PROTOCOL),'--run-id',RUN_ID])
if not shutil.which('prodigal'):
    subprocess.run(['apt-get','update'],check=True); subprocess.run(['apt-get','install','-y','prodigal'],check=True)
run_logged(out,'generation_e100',['seqtrainer-titans-stage-c-generate','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'validation.json'),'--taxonomy-manifest',TAXONOMY_MANIFEST,'--checkpoint',str(checkpoint),'--output-dir',str(out/'generation_t0p6'),'--split','val','--species','Escherichia coli','--prompts','4','--prompt-tokens','128','--new-tokens','1024','--temperatures','0.6','--top-k','1024','--top-p','0.99','--device','cuda','--memory-mode','adaptive','--prodigal',shutil.which('prodigal'),'--protocol',str(PROTOCOL),'--run-id',RUN_ID])
e25=Path(DRIVE_ROOT)/'runs/c19_v3_medium_adaptive_e25/scale_analysis_v1'
subprocess.run(['seqtrainer-titans-stage-c-scale-gate','--stage',STAGE,'--baseline-evaluation',str(e25/'evaluation/evaluation.json'),'--baseline-name','adaptive','--candidate-evaluation',str(out/'evaluation/evaluation.json'),'--candidate-name','adaptive','--memory-behavior',str(out/'memory_behavior.json'),'--baseline-generation',str(e25/'generation_t0p6/generation_evaluation.json'),'--candidate-generation',str(out/'generation_t0p6/generation_evaluation.json'),'--output-dir',str(out/'gate')],check=True)
record_once(RUN_ID,out,'exploratory')
print((out/'gate/SCALE_GATE.md').read_text())
print('A PROCEED result authorizes planning matched no-memory/frozen/second-seed controls; it does not itself prove a memory benefit.')
print('SHARE THIS DIRECTORY:',out)
""",
    ],
    gpu="A100",
)
