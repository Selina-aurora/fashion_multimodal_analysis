from pathlib import Path
import tempfile,csv,json,sys,subprocess
from PIL import Image
from build_existing_train_dev_pool import inspect_inventory,select_review,main as _unused

def run():
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp); proj=base/'fashion_multimodal_analysis'; d=base/'fashion_data';
        (proj/'configs').mkdir(parents=True); train=d/'raw/train/train/image';train.mkdir(parents=True)
        (d/'raw/test/image').mkdir(parents=True)
        Image.new('RGB',(256,256),(130,170,220)).save(train/'img1.jpg')
        Image.new('RGB',(256,256),(70,90,120)).save(train/'img2.jpg')
        Image.new('RGB',(256,256),(150,100,120)).save(d/'raw/test/image/img3.jpg')
        rows=[
          {'record_id':'a','source_dataset':'DeepFashion2','source_image':'../fashion_data/raw/train/train/image/img1.jpg','garment_id':'1','garment_category':'top','bbox_x1':'10','bbox_y1':'20','bbox_x2':'200','bbox_y2':'210'},
          {'record_id':'b','source_dataset':'DeepFashion2','source_image':'../fashion_data/raw/train/train/image/img2.jpg','garment_id':'2','garment_category':'pants','bbox_x1':'5','bbox_y1':'10','bbox_x2':'180','bbox_y2':'180'},
          {'record_id':'c','source_dataset':'DeepFashion2','source_image':'../fashion_data/raw/test/image/img3.jpg','garment_id':'3','garment_category':'dress','bbox_x1':'5','bbox_y1':'10','bbox_x2':'180','bbox_y2':'180'},
          {'record_id':'d','source_dataset':'DeepFashion2','source_image':'../fashion_data/raw/train/train/image/no.jpg','garment_id':'4','garment_category':'bag','bbox_x1':'5','bbox_y1':'10','bbox_x2':'180','bbox_y2':'180'},
          {'record_id':'e','source_dataset':'DeepFashion2','source_image':'../fashion_data/raw/train/train/image/img1.jpg','garment_id':'1','garment_category':'top','bbox_x1':'100','bbox_y1':'50','bbox_x2':'100','bbox_y2':'190'},
        ]
        inv=proj/'configs/prd_8class_unified_manifest_v2.csv'
        with inv.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        audit=inspect_inventory(proj,d,inv)
        assert [r['decision'] for r in audit]==['TRAIN_DEV_ONLY','TRAIN_DEV_ONLY','EXCLUDED','EXCLUDED','EXCLUDED'],audit
        assert len(select_review(audit,2))==2
        assert 'RESTRICTED_PATH' in audit[2]['reason_codes']
        assert 'TRAIN_FILE_NOT_FOUND' in audit[3]['reason_codes']
        output=base/'out'
        subprocess.run([sys.executable,str(Path(__file__).parent/'build_existing_train_dev_pool.py'),'--project-root',str(proj),'--data-root',str(d),'--output',str(output),'--per-class','2'],check=True,capture_output=True,text=True)
        summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
        assert summary['train_dev_paths_present']==2, summary
        assert summary['quarantined_records']==3, summary
        assert summary['preview_created']==2, summary
        html=(output/'TRAIN_DEV_GT_REVIEW_STATIC.html').read_text(encoding='utf-8')
        assert html.count('data-case=')==2 and html.count('data:image/jpeg;base64,')==2
        assert 'TRAIN_DEV_DIAGNOSTIC_ONLY' in html
        print('EXISTING_DATA_TEST_PASS (source paths, restricted exclusion, bboxes, images, HTML)')
if __name__=='__main__':run()
