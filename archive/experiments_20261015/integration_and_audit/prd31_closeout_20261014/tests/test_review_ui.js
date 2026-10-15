// Offline event/geometry/export tests using a minimal DOM harness.
// This is NOT a real-browser rendering test.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'../01_同批样本复核.html'),'utf8');
const data=html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\s*('use strict';[\s\S]*?)<\/script>/)[1];
let elements={},saved={},alerts=[],downloaded;
class El{constructor(tag='div'){this.tag=tag;this.children=[];this.style={};this.value='';this.textContent='';this.checked=false;this.clientWidth=600;this.classList={add:()=>{},remove:()=>{}}}set id(s){this._id=s;elements[s]=this}get id(){return this._id}append(...x){this.children.push(...x)}replaceChildren(...x){this.children=x}click(){if(this.onclick)this.onclick()}setPointerCapture(){}getBoundingClientRect(){return {left:0,top:0,width:500,height:750}}getContext(){return {clearRect(){},drawImage(){},strokeRect(){}}}}
function get(id){return elements[id]||(elements[id]=new El())}get('data').textContent=data;get('zoom').value='100';
class Img{constructor(){this.complete=true;this.naturalWidth=683;this.naturalHeight=1024}set src(v){this._src=v;if(this.onload)this.onload()}get src(){return this._src}}
const ctx=vm.createContext({console,JSON,Date,Math,Number,String,Object,Array,Error,Image:Img,Blob,
 document:{getElementById:get,createElement:t=>new El(t),createTextNode:s=>s},
 localStorage:{getItem:k=>saved[k]||null,setItem:(k,v)=>{saved[k]=v}},
 window:{addEventListener:()=>{}},alert:s=>alerts.push(s),setTimeout:f=>f(),
 URL:{createObjectURL:b=>{downloaded=b;return 'blob:test'},revokeObjectURL:()=>{}}});
vm.runInContext(script,ctx);
const evalJS=s=>vm.runInContext(s,ctx);
(async()=>{
 assert.equal(get('nav').children.length,12);
 assert.equal(evalJS('IDS.filter(k=>state.cases[k].confirmed).length'),0);
 get('confirmed').checked=true;get('confirmed').onchange();assert.equal(get('confirmed').checked,false);
 get('reviewer').value='SYNTHETIC_TEST_ONLY';get('reviewer').oninput();
 get('category').value='top';get('category').onchange();get('targetStatus').value='correct_target';get('targetStatus').onchange();
 for(const k of evalJS('ATTRS')){get('attr_'+k).value=evalJS('DATA.taxonomy['+JSON.stringify(k)+'][0]');get('attr_'+k).onchange()}
 get('presence_waist').value='uncertain';get('presence_waist').onchange();
 get('canvas').onpointerdown({clientX:50,clientY:150,pointerId:1});
 get('canvas').onpointerup({clientX:250,clientY:450,pointerId:1});
 assert.deepEqual(JSON.parse(evalJS('JSON.stringify(rec().target_box)')),[.1,.2,.5,.6]);
 get('confirmed').checked=true;get('confirmed').onchange();assert.equal(evalJS('rec().confirmed'),true);
 get('export').onclick();let obj=JSON.parse(await downloaded.text());assert.equal(obj.cases.case_01.confirmed,true);assert.equal(Object.keys(obj.cases).length,12);
 get('attr_primary_color').value='uncertain';get('attr_primary_color').onchange();assert.equal(evalJS('rec().confirmed'),false);
 get('drawMode').value='cuff';get('canvas').onpointerdown({clientX:30,clientY:30,pointerId:2});get('canvas').onpointerup({clientX:60,clientY:70,pointerId:2});
 assert.equal(evalJS('rec().regions.cuff.boxes.length'),1);get('undo').onclick();assert.equal(evalJS('rec().regions.cuff.boxes.length'),0);
 const old=JSON.parse(evalJS('JSON.stringify(state)'));old.manifest_sha256='wrong';ctx.invalid=old;
 assert.throws(()=>evalJS('validate(invalid)'));
 console.log('UI_LOGIC_PASS: 12 cases; incomplete confirmation blocked; draw coordinates; export; edits invalidate confirmation; undo; identity guard.');
 console.log('LIMIT: DOM harness only; real browser rendering not exercised (no Chromium executable).');
})().catch(e=>{console.error(e);process.exit(1)});
