// LaTeX math -> glyph layout with MathJax 3 (SVG output, no font cache).
// stdin: JSON lines {"id":..., "tex":"..."}; stdout: JSON lines {"id", "ok", "glyphs":[{"c":<codepoint hex or "BAR">,
// "bbox":[x0,y0,x1,y1], "node":<mml node>}], "w", "h", "err"}. Units: 1/1000 em, y grows DOWN (page convention).
// Glyph bbox = extent of the outline's absolute coordinates (MathJax glyph paths use absolute M/L/Q/T commands) under the
// accumulated translate/scale chain; <rect> (fraction bars, radical overbars, table rules) are emitted as c="BAR".
// Usage: node layout.mjs < in.jsonl > out.jsonl
import {mathjax} from 'mathjax-full/js/mathjax.js';
import {TeX} from 'mathjax-full/js/input/tex.js';
import {SVG} from 'mathjax-full/js/output/svg.js';
import {liteAdaptor} from 'mathjax-full/js/adaptors/liteAdaptor.js';
import {RegisterHTMLHandler} from 'mathjax-full/js/handlers/html.js';
import {AllPackages} from 'mathjax-full/js/input/tex/AllPackages.js';
import readline from 'node:readline';
const adaptor=liteAdaptor(); RegisterHTMLHandler(adaptor);
const tex=new TeX({packages:AllPackages, formatError:(jax,err)=>{throw err;}});
const doc=mathjax.document('',{InputJax:tex,OutputJax:new SVG({fontCache:'none'})});
function parseT(s){ // -> affine [a,b,c,d,e,f] (x'=a x + c y + e, y'=b x + d y + f)
  let m=[1,0,0,1,0,0]; if(!s) return m;
  const re=/(translate|scale)\(([^)]*)\)/g; let r;
  while((r=re.exec(s))){ const v=r[2].split(/[ ,]+/).filter(x=>x!=='').map(Number); let t;
    if(r[1]==='translate') t=[1,0,0,1,v[0]||0,v[1]||0]; else t=[v[0],0,0,(v.length>1?v[1]:v[0]),0,0];
    m=mul(m,t); }
  return m; }
function mul(A,B){ return [A[0]*B[0]+A[2]*B[1], A[1]*B[0]+A[3]*B[1], A[0]*B[2]+A[2]*B[3], A[1]*B[2]+A[3]*B[3], A[0]*B[4]+A[2]*B[5]+A[4], A[1]*B[4]+A[3]*B[5]+A[5]]; }
function app(M,x,y){ return [M[0]*x+M[2]*y+M[4], M[1]*x+M[3]*y+M[5]]; }
function boxOf(M,pts){ let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9; for(const [x,y] of pts){ const [u,v]=app(M,x,y); x0=Math.min(x0,u); x1=Math.max(x1,u); y0=Math.min(y0,v); y1=Math.max(y1,v);} return [x0,y0,x1,y1]; }
function pathPoints(d){ // SVG path -> list of points (end points and control points), handles M L T Q C S H V Z, abs and rel
  const tk=d.match(/[A-Za-z]|-?\d*\.?\d+(?:e-?\d+)?/g)||[]; const pts=[]; let cx=0,cy=0,sx=0,sy=0,cmd='M',i=0;
  const n={M:2,L:2,T:2,Q:4,C:6,S:4,H:1,V:1,Z:0,A:7};
  while(i<tk.length){ if(/[A-Za-z]/.test(tk[i])){ cmd=tk[i]; i++; if(cmd==='Z'||cmd==='z'){ cx=sx; cy=sy; continue; } }
    const U=cmd.toUpperCase(), rel=cmd!==U, k=n[U]; if(!k){ i++; continue; }
    const v=tk.slice(i,i+k).map(Number); if(v.length<k) break; i+=k;
    if(U==='H'){ cx=rel?cx+v[0]:v[0]; pts.push([cx,cy]); continue; }
    if(U==='V'){ cy=rel?cy+v[0]:v[0]; pts.push([cx,cy]); continue; }
    if(U==='A'){ cx=rel?cx+v[5]:v[5]; cy=rel?cy+v[6]:v[6]; pts.push([cx,cy]); continue; }
    for(let j=0;j<k;j+=2){ const x=rel?cx+v[j]:v[j], y=rel?cy+v[j+1]:v[j+1]; pts.push([x,y]); if(j===k-2){ cx=x; cy=y; } }
    if(U==='M'){ sx=cx; sy=cy; cmd=rel?'l':'L'; } }
  return pts; }
function walk(node,M,out,mml){
  if(!adaptor.kind || typeof node!=='object') return;
  const kind=adaptor.kind(node); if(kind==='#text'||kind==='#comment') return;
  const t=adaptor.getAttribute(node,'transform'); const M2=t?mul(M,parseT(t)):M;
  const mm=adaptor.getAttribute(node,'data-mml-node')||mml;
  if(kind==='path'){ const c=adaptor.getAttribute(node,'data-c'); const d=adaptor.getAttribute(node,'d')||'';
    const pts=pathPoints(d);
    if(c&&pts.length) out.push({c, bbox:boxOf(M2,pts), node:mm}); return; }
  if(kind==='text' && process.env.LAYOUT_TEXT==='1'){ // unknown-font text (Cyrillic in \\text{}) -> one TEXT box per char:
    // advance 600 (MathJax's estimate, see translate(600k) of the siblings), x-height 450 above the baseline, in the accumulated frame
    const txt=(adaptor.textContent?adaptor.textContent(node):'')||''; const sc=Math.hypot(M2[0],M2[1])||1; const [bx,by]=app(M2,0,0);
    out.push({c:'TEXT', ch:txt, bbox:[bx,by-450*sc,bx+600*sc,by], node:mm}); return; }
  if(kind==='rect'){ const x=+adaptor.getAttribute(node,'x')||0, y=+adaptor.getAttribute(node,'y')||0, w=+adaptor.getAttribute(node,'width')||0, h=+adaptor.getAttribute(node,'height')||0;
    out.push({c:'BAR', bbox:boxOf(M2,[[x,y],[x+w,y+h]]), node:mm}); return; }
  for(const ch of adaptor.childNodes(node)) walk(ch,M2,out,mm);
}
function layout(s){
  const node=doc.convert(s,{display:true}); const svg=adaptor.firstChild(node);
  const out=[]; walk(svg,[1,0,0,1,0,0],out,null);
  // the top <g> flips y (scale(1,-1)), so y already grows down in the accumulated frame
  for(const g of out){ const b=g.bbox; g.bbox=[+b[0].toFixed(1),+Math.min(b[1],b[3]).toFixed(1),+b[2].toFixed(1),+Math.max(b[1],b[3]).toFixed(1)]; }
  const err=adaptor.getAttribute(svg,'data-mjx-error')||null; let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  for(const g of out){ x0=Math.min(x0,g.bbox[0]); y0=Math.min(y0,g.bbox[1]); x1=Math.max(x1,g.bbox[2]); y1=Math.max(y1,g.bbox[3]); }
  return {glyphs:out, w:x1-x0, h:y1-y0, err};
}
const rl=readline.createInterface({input:process.stdin});
for await (const line of rl){ if(!line.trim()) continue; let o; try{o=JSON.parse(line);}catch(e){continue;}
  try{ const r=layout(o.tex); console.log(JSON.stringify({id:o.id, ok:!r.err && r.glyphs.length>0, ...r})); }
  catch(e){ console.log(JSON.stringify({id:o.id, ok:false, err:String(e.message||e).slice(0,200)})); } }
