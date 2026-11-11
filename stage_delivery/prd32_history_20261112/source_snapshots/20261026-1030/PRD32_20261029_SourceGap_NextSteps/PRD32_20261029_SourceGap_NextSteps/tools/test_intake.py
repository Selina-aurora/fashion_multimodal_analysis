import tempfile, json, csv, subprocess, sys
from pathlib import Path
from PIL import Image
script=Path(__file__).with_name('intake_fresh8.py')
with tempfile.TemporaryDirectory() as d:
 root=Path(d)/'fresh_pool';root.mkdir();out=Path(d)/'out';ref=Path(d)/'old.csv'
 with ref.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['source_image']);w.writeheader();w.writerow({'source_image':'/old/top_legacy.jpg'})
 colors=[(140,30,30),(30,140,30),(30,30,140),(180,180,30),(30,160,160),(180,30,160),(60,60,60),(150,110,50)]
 cats=['top','pants','skirt','outerwear','dress','shoe','bag','accessory']
 for cat,color in zip(cats,colors):
  (root/cat).mkdir();Image.new('RGB',(200,250),color).save(root/cat/(cat+'.jpg'))
 # a renamed identical-image should be flagged as duplicate content
 (root/'top'/'top_legacy.jpg').write_bytes((root/'top'/'top.jpg').read_bytes())
 run=subprocess.run([sys.executable,str(script),'--images',str(root),'--output',str(out),'--reference-csv',str(ref)],capture_output=True,text=True)
 assert run.returncode==0,run.stderr
 summary=json.load(open(out/'intake_summary.json'))
 assert summary['images_count']==9 and len(summary['category_counts'])==8,summary
 assert summary['flag_counts']['BASENAME_MATCHES_OLD_DATA']==1,summary
 assert summary['flag_counts']['DUPLICATE_IMAGE_CONTENT']==1,summary
 html=(out/'GT_review_STATIC.html').read_text(encoding='utf-8')
 assert html.count('<article class="card"')==9
 assert html.count('data:image/jpeg;base64,')==9
 assert 'function exportCSV()' in html and 'function exportJSON()' in html
 print('FRESH8_SYNTHETIC_TEST_PASS: 9 records, 8 categories, duplicate+legacy risk detected, static image cards')
