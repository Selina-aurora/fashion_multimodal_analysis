import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import base64

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('collector', ROOT/'collect_remaining_129.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)/'PRD32_DEV60_QWEN_BASELINE_20261030_run01'
        self.marked = self.base/'marked_inputs'
        self.marked.mkdir(parents=True)
        self.rem = self.base/'rem.csv'
        self.work = ROOT/'priority_worklist_171.csv'
        self.full = ROOT/'dev60_171_review_status.csv'
        self.logs = self.base/'qwen_results.jsonl'
        self.outputs = self.base/'collected.zip'
        c.BASE_ROOT = self.base
        c.DEFAULT_MARKED = self.marked
        c.DEFAULT_LOG = self.logs

        fullrows = c.csv_rows(ROOT/'dev60_171_review_status.csv')
        remrows = c.csv_rows(ROOT/'remaining_129_review_queue.csv')
        fake_paths = {}
        for i, row in enumerate(fullrows):
            key = (row['case_id'],row['instance_id'])
            p = self.marked/f'marked_{i:03d}.png'
            # A single-pixel PNG is sufficient: these tests validate file identity, not pixels.
            p.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7u8AAAAASUVORK5CYII='))
            fake_paths[key] = p
        for row in remrows:
            row['marked_baseline_image_path'] = str(fake_paths[(row['case_id'],row['instance_id'])])
        self.rem.write_bytes(c.csv_blob(remrows))
        with self.logs.open('w') as f:
            for row in fullrows:
                key = (row['case_id'],row['instance_id'])
                p = fake_paths[key]
                digest=hashlib.sha256(p.read_bytes()).hexdigest()
                for task in ('tags','vqa','copy'):
                    r={'case_id':key[0],'target_instance_id':key[1],'task':task,
                        'model_variant':'baseline', 'request_id':'__'.join((*key,task)),
                        'image_sha256':row['image_sha256'],
                        'marked_image':str(p),'marked_image_sha256':digest}
                    f.write(json.dumps(r)+'\n')

    def run_collect(self,count=129,out=None):
        return c.collect(remaining=self.rem,worklist=self.work,
           full_status=self.full,marked_root=self.marked,baseline_log=self.logs,
           output=out or self.outputs,count=count)

    def test_all_129(self):
        result=self.run_collect()
        self.assertEqual(result['selected_count'],129)
        self.assertEqual(result['priority_counts'],{'P1_BLOCKED_UNREVIEWED':54,'P2_DRIFT_UNREVIEWED':15,'P3_OTHER_UNREVIEWED':60})
        from zipfile import ZipFile
        with ZipFile(self.outputs) as z:
            self.assertEqual(len([n for n in z.namelist() if n.startswith('images/')]),129)
            self.assertEqual(len(json.loads(z.read('marked_image_manifest.json'))),129)
            self.assertEqual(len(c.csv_rows(self.rem)),129)

    def test_first_30(self):
        out=self.outputs.with_name('30.zip')
        result=self.run_collect(30,out)
        self.assertEqual(result['selected_count'],30)
        self.assertEqual(result['priority_counts'],{'P1_BLOCKED_UNREVIEWED':30})

    def test_no_overwrite(self):
        self.run_collect()
        with self.assertRaises(FileExistsError):self.run_collect()

    def test_bad_marked_digest_blocked(self):
        recs=c.baseline_records(self.logs)
        first=next(iter(c.csv_rows(self.rem)))
        p=Path(first['marked_baseline_image_path']);p.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):self.run_collect()

    def test_missing_task_blocked(self):
        lines=self.logs.read_text().splitlines()
        self.logs.write_text('\n'.join(lines[1:])+'\n')
        with self.assertRaisesRegex(ValueError, 'not full DEV60'):self.run_collect()

    def test_wrong_root_blocked(self):
        r=c.csv_rows(self.rem)
        r[0]['marked_baseline_image_path']='/workspace/HOLDOUT/secret.jpg'
        self.rem.write_bytes(c.csv_blob(r))
        with self.assertRaises(Exception):self.run_collect()

    def test_already_confirmed_blocked(self):
        r=c.csv_rows(self.rem);r[0]['category_GT_status']='CONFIRMED_DETERMINATE'
        self.rem.write_bytes(c.csv_blob(r))
        with self.assertRaisesRegex(ValueError,'confirmed instance'):self.run_collect()

if __name__ == '__main__':unittest.main()
