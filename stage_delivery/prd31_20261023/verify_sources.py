"""Read-only source integrity check; no model loading or accuracy claim."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root, records):
    for relative, expected in records.items():
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('Unsafe relative path: ' + relative)
        if digest(path) != expected:
            raise ValueError('Source SHA256 mismatch: ' + relative)
    return len(records)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=here.parents[1])
    args = parser.parse_args()
    try:
        candidate = here / 'candidate/PRD31_W7_1022_FIRST'
        count = check(candidate, json.loads((candidate / 'PACKAGE_SHA256.json').read_text()))
        sources = {row['path']: row['sha256'] for row in json.loads((here / 'runtime_source_provenance.json').read_text())}
        source_count = check(args.project.resolve(), sources)
        print(json.dumps({'status': 'SOURCE_FILES_OK', 'candidate_files_verified': count,
                          'project_sources_verified': source_count, 'GPU_inference_executed': False}))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({'status': 'SOURCE_FILES_BLOCKED', 'reason': str(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
