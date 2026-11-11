import csv
import hashlib
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[1]/'build_fixed_dev_v2.py'
spec=importlib.util.spec_from_file_location('builddev60',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class TestDev60(unittest.TestCase):
    def test_restricted_paths_and_train_component(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t); data=base/'fashion_data';proj=base/'project';(data/'train').mkdir(parents=True);(data/'holdout').mkdir()
            (data/'train'/'safe.jpg').write_bytes(b'hi')
            (data/'holdout'/'leak.jpg').write_bytes(b'hi')
            p,err=m.safe_train_image(str(data/'train'/'safe.jpg'),proj,data)
            self.assertEqual(err,'');self.assertTrue(p.is_file())
            p,err=m.safe_train_image(str(data/'holdout'/'leak.jpg'),proj,data)
            self.assertIsNone(p)
            self.assertIn(err,('RESTRICTED_PATH','OUTSIDE_TRAIN_ROOT_OR_MISSING'))

    def test_dhash_and_exclusions(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'foo.jpg'
            Image.new('RGB',(96,96),(80,90,100)).save(p)
            self.assertEqual(m.hamming(m.d_hash(p)[0],m.d_hash(p)[0]),0)
            e=Path(t)/'approved_digests.txt'
            e.write_text('\n'.join(hashlib.sha256(bytes([j])).hexdigest() for j in range(8)))
            self.assertEqual(len(m.read_approved_exclusions(e)),8)
            e.write_text('not-a-digest')
            with self.assertRaises(ValueError):m.read_approved_exclusions(e)

    def test_selection_strata_and_duplicate_defense(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);data=base/'fashion_data';proj=base/'project';v1=base/'old_v1';imgdir=data/'train'/'photos'
            imgdir.mkdir(parents=True);proj.mkdir();v1.mkdir()
            all_images=[]
            for i in range(84):
                rng=random.Random(i+1337)
                pix=bytes(rng.randrange(256) for _ in range(60*60*3))
                p=imgdir/f'img_{i:03d}.png'
                Image.frombytes('RGB',(60,60),pix).save(p)
                all_images.append(p)
            old=[]
            for j,p in enumerate(all_images[:12],1):
                old.append({'case_id':f'fresh_dev_{j:03d}','image_path':str(p),'image_sha256':m.sha(p),
                            'target_instance_id':'pred_0000'})
            # 31 historical instances, exact copies still map to the same source image.
            historic=old + [dict(old[i%12],target_instance_id=f'pred_{(i//12)+1:04d}') for i in range(19)]
            m.writejl(v1/'frozen_v5_instances.jsonl',historic)
            m.writej(v1/'freeze.json',{'schema':'prd32_20261030_fixed_dev_freeze_v1','images':12,
               'instances':31,'frozen_instance_manifest_sha256':m.sha(v1/'frozen_v5_instances.jsonl')})
            inv=proj/'inventory.csv'
            with inv.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['source_image','garment_category'])
                w.writeheader()
                for j,p in enumerate(all_images[12:]):
                    w.writerow({'source_image':str(p),'garment_category':m.CLASS_ORDER[j%8]})
            import argparse
            out=base/'out'
            args=argparse.Namespace(output=out,v1=v1,project_root=proj,data_root=data,inventory=inv,
                 exclusion_hashes=None,max_per_class=900,dhash_distance=5,total=60)
            x=m.build(args)
            self.assertEqual(x['status'],'PROVISIONAL_HOLDOUT_EXCLUSION_NOT_VERIFIED')
            self.assertEqual(x['images'],60)
            self.assertEqual(m.verify(args)['new_images'],48)
            m.export(args)
            self.assertTrue((out/'PRD32_DEV60_STAGE1_REVIEW.zip').is_file())
            selected=m.readjl(out/'v5_image_only_inputs_60.jsonl')
            self.assertEqual(len({r['image_sha256'] for r in selected}),60)
            self.assertEqual(len(selected),60)
            for r in selected:
                self.assertNotIn('category',str(r))
            with self.assertRaises(FileExistsError):m.build(args)
            (out/'v5_image_only_inputs_60.jsonl').write_text('changed')
            with self.assertRaises(ValueError):m.verify(args)

if __name__=='__main__':unittest.main()
