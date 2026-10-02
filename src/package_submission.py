"""Package reports, runnable code and saved evidence; never train or evaluate."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def selected_files():
    files = [ROOT / name for name in (
        'README.md', 'requirements.in', 'requirements.txt', '.python-version', 'data/README.md')]
    files += sorted((ROOT / 'docs').glob('*.md'))
    files += sorted((ROOT / 'src').glob('*.py'))
    files += sorted((ROOT / 'notebooks').glob('*.ipynb'))
    files += [ROOT / 'outputs/final/DAY1_REPORT.md']
    files += sorted((ROOT / 'outputs/final/figures').glob('*.png'))
    allowed = {'.md', '.png', '.csv', '.json', '.joblib'}
    files += sorted(p for p in (ROOT / 'outputs/day2').rglob('*')
                    if p.is_file() and p.suffix in allowed)
    return sorted(set(files))


def main():
    files = selected_files()
    if any(not p.is_file() for p in files):
        raise FileNotFoundError('Required submission files are missing')
    hashes = []
    target = ROOT / 'submission.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data/', b'')
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            payload = path.read_bytes()
            archive.writestr(name, payload)
            hashes.append(f'{hashlib.sha256(payload).hexdigest()}  {name}')
        archive.writestr('CONTENTS.sha256', '\n'.join(hashes) + '\n')
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Submission archive integrity check failed')
    print(f'{target.name}: {len(files)} files, {target.stat().st_size / 1024**2:.2f} MiB')


if __name__ == '__main__':
    main()
