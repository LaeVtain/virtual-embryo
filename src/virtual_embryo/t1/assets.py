"""Export the five frozen predictors and their continuation-training assets."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
from ..paths import task_assets
from .model import checkpoint_config, inference_state


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def export_assets(workspace: Path, destination: Path) -> dict:
    workspace, destination = Path(workspace), Path(destination)
    source = workspace/'T1_joint_moment_clean_20261112'
    destination.mkdir(parents=True, exist_ok=True)
    for part in ('inference', 'training', 'prepared'):
        (destination/part).mkdir(exist_ok=True)
    manifest = dict(teacher_seeds=[11, 22, 33], noise_offsets=[0, 10000, 20000],
                    source_sha256={}, exported_sha256={}, inference_tensor_bytes=0)
    for name in ('original', 'fullELBO30', 'teacher11', 'teacher22', 'teacher33'):
        path = source/'weights'/f'{name}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if name == 'original':
            manifest['config'] = checkpoint_config(checkpoint)
            (destination/'control_initialization.json').write_text(
                json.dumps(checkpoint['record'], indent=2), encoding='utf-8')
        state = checkpoint['model']
        compact = inference_state(state)
        manifest['inference_tensor_bytes'] += sum(v.numel()*v.element_size() for v in compact.values())
        torch.save(dict(model=compact), destination/'inference'/f'{name}.pt')
        if name in ('original', 'fullELBO30'):
            torch.save(dict(model=state, config=manifest['config']), destination/'training'/f'{name}.pt')
        manifest['source_sha256'][name] = digest(path)
        del checkpoint, state, compact
    shutil.copyfile(source/'genes.json', destination/'genes.json')
    prepared = workspace/'T1_prior_detection/prepared'
    for name in ('probe.joblib', 'coupling_0_1.json', 'coupling_1_2.json', 'coupling_0_2.json',
                 'pairs_0_1.npz', 'pairs_1_2.npz', 'pairs_0_2.npz'):
        shutil.copyfile(prepared/name, destination/'prepared'/name)
    for path in sorted(destination.rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            manifest['exported_sha256'][path.relative_to(destination).as_posix()] = digest(path)
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--workspace', type=Path)
    source.add_argument('--trained-run', type=Path, help='Output directory of train all')
    parser.add_argument('--base-model-dir', type=Path, default=task_assets('t1'))
    parser.add_argument('--destination', type=Path, default=task_assets('t1'))
    args = parser.parse_args()
    result = (export_assets(args.workspace, args.destination) if args.workspace is not None
              else assemble_training_run(args.trained_run, args.base_model_dir, args.destination))
    print(json.dumps(dict(destination=str(args.destination), files=len(result['exported_sha256'])), indent=2))


def assemble_training_run(run_dir: Path, base_model_dir: Path, destination: Path) -> dict:
    """Package an `all` training run as a separately selectable predictor."""
    run_dir, base_model_dir, destination = map(Path, (run_dir, base_model_dir, destination))
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'inference').mkdir()
    base = json.loads((base_model_dir/'manifest.json').read_text(encoding='utf-8'))
    manifest = {key: base[key] for key in ('config', 'teacher_seeds', 'noise_offsets')}
    manifest.update(source_sha256={}, exported_sha256={})
    for name in ('original', 'fullELBO30', 'teacher11', 'teacher22', 'teacher33'):
        source = (base_model_dir/'inference/original.pt' if name == 'original'
                  else run_dir/name/'inference.pt')
        target = destination/'inference'/f'{name}.pt'
        shutil.copyfile(source, target)
        manifest['source_sha256'][name] = digest(source)
        manifest['exported_sha256'][target.relative_to(destination).as_posix()] = digest(target)
    shutil.copyfile(base_model_dir/'genes.json', destination/'genes.json')
    manifest['exported_sha256']['genes.json'] = digest(destination/'genes.json')
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    main()
