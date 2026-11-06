# -*- coding: utf-8 -*-
"""Scrawler tracer: image of ONE handwritten line -> pen trajectory (points, pen lifts, pressure).

Pipeline: crop -> 64 px high frame -> ink restorer (U-Net) -> rule removal -> [optional neighbour-line ink removal] -> islands ->
autoregressive tracer per island (sampling, seeded) -> guard -> second pass over untraced own ink -> stroke merge.
Known limits: strokes of the second pass are appended at the end of the trace, so the stroke ORDER of that part is not the writing order.
erase_neighbours=True cuts pick-ups from the lines above/below on tight prose crops, but it also deletes own ink of multi-level layouts
(fractions, roots, formulas) and costs recall on the benchmark, so it is off by default.

Usage:  python scrawler_trace.py line.png --out trace.json [--svg trace.svg] [--seed 0] [--erase_neighbours]
API:    tr = Tracer('weights'); r = tr.trace_image(gray_uint8_array)   # r['px'] (n,2) in input-crop pixels, r['pen'] (n,) 1 = pen lifts after the point, r['pres'] (n,)
Needs:  torch, numpy, pillow, opencv-python. Device: cuda > mps > cpu."""
import os, math, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from PIL import Image

MAXPTS = 1200; SAMPLE_BIAS = 2.0; PENW = 5.0; PEN_TH = PENW / (1.0 + PENW)
DERULE = 0.5; SPLIT_LONG = 6; GUARD_W = 1.5; GUARD_K = 6; BRIDGE = 1.5; MERGE = 1.5; MERGE_COS = 0.5; IBS = 48; TRACE_ML_K = 3.0; RES_R = 1.5; RES_MIN = 40
CORE_FRAC = 0.35; CORE_MIN = 0.15; FLAT_H = 0.3; NB_FAR = 1.2

# ---------------------------------------------------------------- model (tracer)
def strokes_of(pen):
    """pen [N] (1 = pen lifts AFTER this point) -> list of (a,b) inclusive index ranges."""
    N=len(pen); ends=list(np.where(pen>0.5)[0])
    if not ends or ends[-1]!=N-1: ends.append(N-1)
    out=[]; a=0
    for e in ends:
        if e>=a: out.append((a,int(e)))
        a=int(e)+1
    return out
def norm_img(g):
    """uint8 gray [H,W] (dark ink on light paper) -> float32, paper ~0, ink ~1. Same at train time and on real scans."""
    x=1.0-np.asarray(g,np.float32)/255.0; med=float(np.median(x)); hi=float(np.percentile(x,99.5)); x=(x-med)/max(hi-med,0.15)
    return np.clip(x,-1.0,1.5).astype(np.float32)
def fourier(p,nf):
    """p [...,2] in image-height units -> [..., 4*nf]; wavelengths 32 .. 0.05 (x spans up to MAXW/IMH = 16)."""
    wl=torch.logspace(math.log10(32.0),math.log10(0.05),nf).to(p.device); a=p[...,:,None].float()*(2*math.pi/wl)
    return torch.cat([a.sin(),a.cos()],-1).flatten(-2)
def step_pe(t,d):
    """sinusoidal encoding of step indices t [...] -> [..., d]."""
    i=torch.arange(d//2,device=t.device).float(); a=t[...,None].float()/(10000**(2*i/d))
    return torch.cat([a.sin(),a.cos()],-1)
class Block(nn.Module):
    def __init__(s,ci,co,st):
        super().__init__(); s.c1=nn.Conv2d(ci,co,3,st,1,bias=False); s.n1=nn.GroupNorm(8,co); s.c2=nn.Conv2d(co,co,3,1,1,bias=False); s.n2=nn.GroupNorm(8,co)
        s.sc=nn.Sequential(nn.Conv2d(ci,co,1,st,bias=False),nn.GroupNorm(8,co)) if (st!=1 or ci!=co) else nn.Identity()
    def forward(s,x): return F.relu(s.sc(x)+s.n2(s.c2(F.relu(s.n1(s.c1(x))))))
class CNN(nn.Module):
    """GroupNorm (no running stats -> EMA of parameters is a complete EMA model). Returns stride-4 and stride-8 maps."""
    def __init__(s,ch):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv2d(1,32,3,1,1,bias=False),nn.GroupNorm(8,32),nn.ReLU(),nn.Conv2d(32,ch,3,2,1,bias=False),nn.GroupNorm(8,ch),nn.ReLU())
        s.l1=nn.Sequential(Block(ch,ch,1),Block(ch,ch,1)); s.l2=nn.Sequential(Block(ch,2*ch,2),Block(2*ch,2*ch,1))
        s.l3=nn.Sequential(Block(2*ch,4*ch,2),Block(4*ch,4*ch,1),Block(4*ch,4*ch,1),Block(4*ch,4*ch,1))
    def forward(s,x): f4=s.l2(s.l1(s.stem(x))); return f4,s.l3(f4)
class DecLayer(nn.Module):
    def __init__(s,d,h,drop):
        super().__init__(); s.h=h; s.n1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.o1=nn.Linear(d,d); s.n2=nn.LayerNorm(d); s.q2=nn.Linear(d,d); s.kv2=nn.Linear(d,2*d); s.o2=nn.Linear(d,d)
        s.n3=nn.LayerNorm(d); s.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)); s.dr=nn.Dropout(drop)
    def mem(s,m):
        Bn,K,d=m.shape; k,v=s.kv2(m).view(Bn,K,2,s.h,d//s.h).permute(2,0,3,1,4); return k,v
    def forward(s,x,mkv,mmask,cache=None):
        Bn,T,d=x.shape; q,k,v=s.qkv(s.n1(x)).view(Bn,T,3,s.h,d//s.h).permute(2,0,3,1,4)
        if cache is not None:
            if 'k' in cache: k=torch.cat([cache['k'],k],2); v=torch.cat([cache['v'],v],2)
            cache['k']=k; cache['v']=v; a=F.scaled_dot_product_attention(q,k,v)
        else: a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+s.dr(s.o1(a.transpose(1,2).reshape(Bn,T,d)))
        q2=s.q2(s.n2(x)).view(Bn,T,s.h,d//s.h).transpose(1,2); a2=F.scaled_dot_product_attention(q2,mkv[0],mkv[1],attn_mask=mmask)
        x=x+s.dr(s.o2(a2.transpose(1,2).reshape(Bn,T,d)))
        return x+s.dr(s.ff(s.n3(x)))
class Model(nn.Module):
    def __init__(s,cfg):
        super().__init__(); s.cfg=dict(cfg); d=cfg['DM']; ch=cfg['CH']; s.nf=cfg['NF']; s.imh=cfg['IMH']; s.M=cfg['M']; s.pres=cfg['PRES']
        s.cnn=CNN(ch); s.proj=nn.Conv2d(4*ch,d,1); s.kpos=nn.Linear(4*s.nf,d)
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,cfg['NH'],4*d,cfg['DROP'],batch_first=True,norm_first=True),cfg['NE'],enable_nested_tensor=False); s.enorm=nn.LayerNorm(d)
        s.gproj=nn.Conv2d(2*ch,cfg['GC'],1); gs=cfg['GSP']; s.register_buffer('goff',torch.tensor([[dx,dy] for dy in (-gs,0,gs) for dx in (-gs,0,gs)],dtype=torch.float32),persistent=False)
        s.inp=nn.Linear(3,d); s.qpos=nn.Linear(4*s.nf,d); s.glin=nn.Linear(9*cfg['GC'],d); s.start=nn.Parameter(torch.randn(d)*0.02)
        s.dec=nn.ModuleList([DecLayer(d,cfg['NH'],cfg['DROP']) for _ in range(cfg['NL'])]); s.dnorm=nn.LayerNorm(d)
        s.head=nn.Linear(d,6*s.M+2+(1 if s.pres else 0)); s.shead=nn.Linear(d,6*s.M)
    def encode(s,img,wc):
        f4,f8=s.cnn(img); m=s.proj(f8); Bn,d,h8,w8=m.shape
        ys,xs=torch.meshgrid(torch.arange(h8,device=img.device),torch.arange(w8,device=img.device),indexing='ij')
        pos=torch.stack([(xs.flatten()+0.5)*8/s.imh,(ys.flatten()+0.5)*8/s.imh],-1)
        m=m.flatten(2).transpose(1,2)+s.kpos(fourier(pos,s.nf))[None]
        valid=(xs.flatten()[None,:]<wc[:,None])
        m=s.enorm(s.enc(m,src_key_padding_mask=~valid))
        return m,valid[:,None,None,:],s.gproj(f4)
    def glimpse(s,g4,Pa):
        Bn,C,h4,w4=g4.shape; px=Pa*s.imh; g=px[:,:,None,:]+s.goff[None,None]
        grid=torch.stack([g[...,0]/(w4*4)*2-1,g[...,1]/(h4*4)*2-1],-1)
        out=F.grid_sample(g4.float(),grid.float(),align_corners=False)
        return out.permute(0,2,3,1).reshape(Bn,Pa.size(1),9*C)
    def embed(s,Xin,Pa,g4,t0=0):
        Bn,T,_=Xin.shape; t=torch.arange(t0,t0+T,device=Xin.device)
        e=s.inp(Xin)+s.qpos(fourier(Pa,s.nf))+s.glin(s.glimpse(g4,Pa).to(Xin.dtype))
        e=torch.where((t==0)[None,:,None],s.start[None,None].to(e.dtype),e)
        return e+step_pe(t,e.size(-1))[None].to(e.dtype)
    def forward(s,img,wc,Xin,Pa):
        m,mm,g4=s.encode(img,wc); h=s.embed(Xin,Pa,g4)
        for L in s.dec: h=L(h,L.mem(m),mm)
        h=s.dnorm(h); return s.head(h),s.shead(h[:,0])
def mdn_draw(o,M,bias,greedy=False):
    """o [B,6M] -> (dx,dy) [B,2] with the generators' low-temperature bias (pi sharpened by 1+bias, sigma / e^bias)."""
    Bn=o.size(0); ar=torch.arange(Bn,device=o.device)
    pi=F.softmax(o[:,:M]*(1+bias),-1); mu=o[:,M:3*M].view(Bn,M,2); ls=o[:,3*M:5*M].view(Bn,M,2).clamp(-7,5)-bias; rho=torch.tanh(o[:,5*M:6*M])
    k=pi.argmax(-1) if greedy else torch.multinomial(pi,1)[:,0]; m=mu[ar,k]
    if greedy: return m
    sd=ls[ar,k].exp(); r=rho[ar,k]; e1=torch.randn(Bn,device=o.device); e2=torch.randn(Bn,device=o.device)
    return torch.stack([m[:,0]+sd[:,0]*e1,m[:,1]+sd[:,1]*(r*e1+torch.sqrt(1-r**2)*e2)],-1)
@torch.no_grad()
def trace(model,img,wc,bias=SAMPLE_BIAS,maxlen=MAXPTS,greedy=False,pen_th=PEN_TH):
    """img [B,1,H,W] normalised (norm_img), wc [B] valid stride-8 columns -> list of dict(absu (n,2) image units, pen (n,), pres (n,)|None)."""
    was=model.training; model.eval(); dv=img.device; Bn=img.size(0); M_=model.M; SC=model.cfg['SCALE']
    m,mm,g4=model.encode(img,wc); mkv=[L.mem(m) for L in model.dec]; caches=[{} for _ in model.dec]
    def run(xin,pa,t):
        h=model.embed(xin,pa,g4,t)
        for L,kv,c in zip(model.dec,mkv,caches): h=L(h,kv,mm,c)
        h=model.dnorm(h); return model.head(h)[:,-1],h[:,-1]
    o,h0=run(torch.zeros(Bn,1,3,device=dv),torch.zeros(Bn,1,2,device=dv),0)
    p=mdn_draw(model.shead(h0[:,None])[:,0],M_,bias,greedy)
    pts=[p]; pens=[(torch.sigmoid(o[:,6*M_])>pen_th).float()]; prs=[torch.sigmoid(o[:,6*M_+2])] if model.pres else None
    done=torch.sigmoid(o[:,6*M_+1])>pen_th; stop=torch.where(done,torch.ones(Bn,dtype=torch.long,device=dv),torch.zeros(Bn,dtype=torch.long,device=dv))
    prev=torch.zeros(Bn,2,device=dv)
    for t in range(1,maxlen):
        if bool(done.all()): break
        xin=torch.cat([prev*SC,pens[-1][:,None]],-1)[:,None]
        o,_=run(xin,pts[-1][:,None],t)
        dl=mdn_draw(o,M_,bias,greedy)/SC; p=pts[-1]+dl; prev=dl
        pens.append((torch.sigmoid(o[:,6*M_])>pen_th).float()); pts.append(p)
        if prs is not None: prs.append(torch.sigmoid(o[:,6*M_+2]))
        eos=(torch.sigmoid(o[:,6*M_+1])>pen_th)&~done; stop[eos]=t+1; done=done|eos
    stop[stop==0]=len(pts)
    P=torch.stack(pts,1).float().cpu().numpy(); Pn=torch.stack(pens,1).cpu().numpy(); Pr=torch.stack(prs,1).float().cpu().numpy() if prs is not None else None
    if was: model.train()
    out=[]
    for b in range(Bn):
        n=int(stop[b]); pn=Pn[b,:n].copy(); pn[-1]=1; out.append(dict(absu=P[b,:n],pen=pn,pres=(Pr[b,:n] if Pr is not None else None)))
    return out

# ---------------------------------------------------------------- ink restorer
class Blk(nn.Module):
    def __init__(s,c,g=8):
        super().__init__(); s.n1=nn.GroupNorm(g,c); s.c1=nn.Conv2d(c,c,3,padding=1); s.n2=nn.GroupNorm(g,c); s.c2=nn.Conv2d(c,c,3,padding=1)
    def forward(s,x): h=s.c1(F.silu(s.n1(x))); h=s.c2(F.silu(s.n2(h))); return x+h
class UNet(nn.Module):
    def __init__(s,ch=32,nblk=2):
        super().__init__(); C=[ch,2*ch,4*ch,8*ch]; s.cfg=dict(ch=ch,nblk=nblk)
        s.stem=nn.Conv2d(1,C[0],3,padding=1)
        s.enc=nn.ModuleList([nn.Sequential(*[Blk(c) for _ in range(nblk)]) for c in C])
        s.down=nn.ModuleList([nn.Conv2d(C[i],C[i+1],3,stride=2,padding=1) for i in range(3)])
        s.mid=nn.Sequential(Blk(C[3]),nn.Conv2d(C[3],C[3],3,padding=2,dilation=2),nn.SiLU(),Blk(C[3]))
        s.up=nn.ModuleList([nn.ConvTranspose2d(C[i+1],C[i],2,stride=2) for i in range(3)])
        s.fuse=nn.ModuleList([nn.Conv2d(2*C[i],C[i],1) for i in range(3)])
        s.dec=nn.ModuleList([nn.Sequential(*[Blk(c) for _ in range(nblk)]) for c in C[:3]])
        s.head=nn.Sequential(nn.GroupNorm(8,C[0]),nn.SiLU(),nn.Conv2d(C[0],1,3,padding=1))
    def forward(s,x):
        h=s.stem(x); skips=[]
        for i in range(3): h=s.enc[i](h); skips.append(h); h=s.down[i](h)
        h=s.enc[3](h); h=s.mid(h)
        for i in (2,1,0): h=s.up[i](h); h=s.fuse[i](torch.cat([h,skips[i]],1)); h=s.dec[i](h)
        return s.head(h)

# ---------------------------------------------------------------- image helpers
def dilate(mask,r):
    from PIL import Image, ImageFilter
    if r<1: return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8)*255,'L').filter(ImageFilter.MaxFilter(2*int(r)+1)))>127
def skeletonize(mask):
    """Zhang-Suen thinning, numpy."""
    m=np.pad(mask.astype(np.uint8),1)
    while True:
        changed=False
        for it in (0,1):
            P2=m[:-2,1:-1];P3=m[:-2,2:];P4=m[1:-1,2:];P5=m[2:,2:];P6=m[2:,1:-1];P7=m[2:,:-2];P8=m[1:-1,:-2];P9=m[:-2,:-2]
            nb=[P2,P3,P4,P5,P6,P7,P8,P9]; Bc=sum(x.astype(np.int32) for x in nb); seq=nb+[P2]
            A=sum(((seq[k]==0)&(seq[k+1]==1)).astype(np.int32) for k in range(8))
            cond=((P2*P4*P6)==0)&((P4*P6*P8)==0) if it==0 else ((P2*P4*P8)==0)&((P2*P6*P8)==0)
            rm=(m[1:-1,1:-1]==1)&(Bc>=2)&(Bc<=6)&(A==1)&cond
            if rm.any(): m[1:-1,1:-1][rm]=0; changed=True
        if not changed: break
    return m[1:-1,1:-1].astype(bool)
def skel_len(sk):
    p=np.pad(sk.astype(np.uint8),1); c=p[1:-1,1:-1]; R=p[1:-1,2:]; L=p[1:-1,:-2]; Dn=p[2:,1:-1]; DR=p[2:,2:]; DL=p[2:,:-2]
    return float((c&R).sum()+(c&Dn).sum()+math.sqrt(2)*((c&DR&(1-R)&(1-Dn)).sum()+(c&DL&(1-L)&(1-Dn)).sum()))
def to_input(crop,imh,maxw):
    from PIL import Image
    h,w=crop.shape; W=max(8,int(round(w*imh/h))); im=Image.fromarray(crop,'L')
    if W<=maxw: return np.asarray(im.resize((W,imh),Image.BILINEAR))
    H2=max(8,int(round(h*maxw/w))); small=np.asarray(im.resize((maxw,H2),Image.BILINEAR)); g=np.full((imh,maxw),int(np.median(crop)),np.uint8); y=(imh-H2)//2; g[y:y+H2]=small; return g
def _runlen(B,axis):
    """length of the run of ink each pixel belongs to, along axis 0 (vertical runs) or 1 (horizontal runs)."""
    A=(B if axis==0 else B.T).astype(np.int32); H=A.shape[0]; up=np.zeros(A.shape,np.int32); dn=np.zeros(A.shape,np.int32)
    for y in range(H): up[y]=(up[y-1]+1)*A[y] if y else A[y]
    for y in range(H-1,-1,-1): dn[y]=(dn[y+1]+1)*A[y] if y<H-1 else A[y]
    r=(up+dn-1)*A; return r if axis==0 else r.T
def _shear(Tm,off):
    """S[i,j] = Tm[i+off[j], j] (0 outside)."""
    n=Tm.shape[0]; S=np.zeros_like(Tm)
    for o in np.unique(off):
        m=np.where(off==o)[0]; src=np.arange(n)+o; ok=(src>=0)&(src<n); S[np.ix_(np.where(ok)[0],m)]=Tm[np.ix_(src[ok],m)]
    return S
def rule_mask_full(G,N,hsharp=0.2,gapmin=0.5,vsharp=0.3,vcov=0.8,maxth=3,d=3):
    """ruled / grid lines over the WHOLE line frame, found on the GREY image: a faint rule mostly vanishes in the
    binarised mask, but the tracer sees the grey crop and follows it (the binarised rule was
    3% of the line, the traced one ran along all of it). N = norm_img(frame) (paper 0, ink 1), G = binarised ink {0,1}.
    Horizontal: for slopes in +-0.04 shear N so that a line of that slope is a row; row profile = MEDIAN over columns (text covers
    < half of the columns of any row, a rule all of them); candidate = sharp peak prof[y] - max(prof[y-d], prof[y+d]) >= hsharp;
    accepted only if it is dark (>0.2) in >= gapmin of the WORD-GAP columns (no ink away from the band): a dense cursive baseline
    makes a peak too, but is paper in the gaps. Band = rows around the peak with sharpness >= 0.4 peak (<= 2 px each side).
    Vertical (grid, margin, page edge): the same on columns, slopes +-0.08, peak >= vsharp AND dark (>0.2) over >= vcov of the height.
    Mask = band pixels EXCEPT letter ink crossing the line (binarised pixels whose run across the line is > maxth+1 px).
    -> (mask bool [H,W] = pixels to repaint as paper, info dict(h=[(row,slope,sharp,gapfrac)], v=[...]))."""
    G=(np.asarray(G)>0).astype(np.uint8); N=np.asarray(N,np.float32); H,W=G.shape; out=np.zeros((H,W),bool); info=dict(h=[],v=[])
    def peaks(M,slopes,thr):
        n,length=M.shape; best=None; pos=np.arange(length)
        for sl in slopes:
            off=np.round(sl*(pos-0.5*length)).astype(int); prof=np.median(_shear(M,off),axis=1)
            pp=np.pad(prof,d,mode='edge'); sh=prof-np.maximum(pp[:-2*d],pp[2*d:])
            if best is None or sh.max()>best[0]: best=(float(sh.max()),float(sl),sh,off)
        if best is None or best[0]<thr: return [],None,None
        _,sl,sh,off=best; rows=[]
        for y in np.argsort(sh)[::-1]:
            if sh[y]<thr: break
            if all(abs(int(y)-r)>2 for r,_,_,_,_ in rows):
                a=int(y); b=int(y)
                while a-1>=0 and a>y-2 and sh[a-1]>=0.4*sh[y]: a-=1
                while b+1<n and b<y+2 and sh[b+1]>=0.4*sh[y]: b+=1
                rows.append((int(y),a,b,sl,float(sh[y])))
        return rows,off,sh
    hr,off,_=peaks(N,np.linspace(-0.04,0.04,17),hsharp)
    if hr:
        Gs=_shear(G,off); Ns=_shear(N,off); letter=(_runlen(G,0)>maxth+1)&(G>0)
        cols=np.where(G.any(0))[0]; x0,x1=(int(cols[0]),int(cols[-1])) if len(cols) else (0,W-1)
        for y0,a,b,sl,shp in hr:
            outside=Gs.copy(); outside[max(0,a-2):min(H,b+3)]=0; empty=np.where(~outside.any(0))[0]; empty=empty[(empty>=x0)&(empty<=x1)]
            if len(empty)<8:
                gapfrac=float('nan')
                if shp<2*hsharp: continue
            else:
                gapfrac=float((Ns[a:b+1,empty].max(0)>0.2).mean())
                if gapfrac<gapmin: continue
            for x in range(W):
                ys=np.arange(a,b+1)+off[x]; ys=ys[(ys>=0)&(ys<H)]; out[ys,x]|=~letter[ys,x]
            info['h'].append((y0,round(sl,3),round(shp,2),round(gapfrac,2)))
    vr,offv,_=peaks(np.ascontiguousarray(N.T),np.linspace(-0.08,0.08,17),vsharp)
    if vr:
        NT=np.ascontiguousarray(N.T); letter=(_runlen(G,1)>maxth+1)&(G>0)
        for x0,a,b,sl,shp in vr:
            dark=np.zeros(H,bool); xs_all=[]
            for y in range(H):
                xs=np.arange(a,b+1)+offv[y]; xs=xs[(xs>=0)&(xs<W)]; xs_all.append(xs); dark[y]=(NT[xs,y]>0.2).any() if len(xs) else False
            if dark.mean()<vcov: continue
            for y,xs in enumerate(xs_all): out[y,xs]|=~letter[y,xs]
            info['v'].append((x0,round(sl,3),round(shp,2)))
    return out,info
def clean_rules(L):
    """repaint the detected rule / grid pixels of the grey frame L with the local paper level (median of the frame)."""
    N=norm_img(L); m,info=rule_mask_full(N>0.5,N)
    if not m.any(): return L,info
    L2=L.copy(); L2[m]=int(np.median(L)); return L2,info
def split_long_units(units,ink,nx):
    """islands wider than nx x-heights (long solid words: the decoder ends early or loses its way on them) are cut into
    pieces <= nx x-heights at the THINNEST column (fewest ink pixels) in the last 40% of each piece; each piece is traced on its own
    (the neighbour piece is painted out like any other island) and --merge_cross glues the strokes back across the cut.
    x-height = FWHM of the row profile of the kept ink."""
    if nx<=0 or not units: return units
    r=np.convolve(ink.sum(1).astype(np.float64),np.ones(3)/3,'same')
    if r.max()<=0: return units
    k=int(r.argmax()); lo=hi=k
    while lo>0 and r[lo-1]>=0.5*r[k]: lo-=1
    while hi<len(r)-1 and r[hi+1]>=0.5*r[k]: hi+=1
    xh=float(np.clip(hi-lo+1,4,0.6*ink.shape[0])); Lmax=nx*xh; out=[]
    for u in units:
        if u['x1']-u['x0']<=Lmax: out.append(u); continue
        m=u['mask']; col=m.sum(0); cuts=[u['x0']]; s=u['x0']
        while u['x1']-s>Lmax:
            a=int(s+0.6*Lmax); b=int(min(u['x1']-0.3*Lmax,s+Lmax))
            if b<=a: break
            seg=col[a:b+1]; c=a+int(np.flatnonzero(seg==seg.min())[-1]); cuts.append(c); s=c
        cuts.append(u['x1'])
        for c0,c1 in zip(cuts[:-1],cuts[1:]):
            pm=np.zeros_like(m); pm[:,c0:c1]=m[:,c0:c1]
            if not pm.any(): continue
            ys,xs=np.nonzero(pm); out.append(dict(x0=int(xs.min()),x1=int(xs.max())+1,y0=int(ys.min()),y1=int(ys.max())+1,mask=pm,split=True))
    out.sort(key=lambda v:(v['x0'],v['x1']))
    return out
def find_islands(L,bridge=0.0):
    """L uint8 line frame, height imh. Binarise as the metric does (norm_img > 0.5), drop non-ink (thick blobs, rules, page
    edges, specks; logic of hwr_pages.clean_components), merge x-overlapping components (parts of one letter) and glue small
    pieces (dots, ticks) to a neighbour closer than 0.15 line height. -> (units: list of dict(x0,x1,y0,y1,mask) left->right,
    ink mask kept, dropped mask, line height hb)."""
    import cv2
    G=(norm_img(L)>0.5).astype(np.uint8); H,W=G.shape; hb=H/1.3
    r=int(max(3,round(0.2*hb))); thick=cv2.morphologyEx(G,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))
    drop=np.zeros_like(G)
    if thick.any():
        nb,lb,sb,_=cv2.connectedComponentsWithStats(thick,8)
        for i in range(1,nb):
            if sb[i,cv2.CC_STAT_AREA]>=(0.3*hb)**2: drop[lb==i]=1
        if drop.any(): drop=cv2.dilate(drop,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))&G
    if DERULE>0:
        Lh=max(8,int(round(DERULE*hb))); Lv=max(8,int(round(1.1*hb)))
        rl=cv2.morphologyEx(G,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(Lh,1)))|cv2.morphologyEx(G,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,Lv)))
        drop|=rl
    ink=G&(1-drop); n,lab,st,_=cv2.connectedComponentsWithStats(ink,8)
    a=st[:,cv2.CC_STAT_AREA].astype(np.float64); x=st[:,0]; y=st[:,1]; w=st[:,2]; h=st[:,3]
    speck=a<4
    linear=((w>0.4*W)&(h<0.25*hb))|((h>=0.85*H)&(w<0.12*H)&(y<=1)&(y+h>=H-1))
    bad=speck|linear; bad[0]=True
    for i in np.where(bad[1:])[0]+1:
        if not speck[i]: drop[lab==i]=1
    keep=[i for i in range(1,n) if not bad[i]]
    par={i:i for i in keep}
    if bridge>0 and len(keep)>1:
        r=max(1,int(math.ceil(bridge/2))); km=np.isin(lab,keep).astype(np.uint8)
        _,ld=cv2.connectedComponents(cv2.dilate(km,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))),connectivity=8); first={}
        for i in keep:
            yy,xx=np.nonzero(lab==i); gid=int(ld[yy[0],xx[0]])
            if gid in first: par[i]=first[gid]
            else: first[gid]=i
    def f(i):
        while par[i]!=i: par[i]=par[par[i]]; i=par[i]
        return i
    def box(gr):
        return min(x[i] for i in gr),max(x[i]+w[i] for i in gr),min(y[i] for i in gr),max(y[i]+h[i] for i in gr),sum(a[i] for i in gr)
    for _ in range(4):
        groups={}
        for i in keep: groups.setdefault(f(i),[]).append(i)
        B={g:box(v) for g,v in groups.items()}; gs=sorted(B,key=lambda g:B[g][0]); changed=False
        for ii,g1 in enumerate(gs):
            for g2 in gs[ii+1:]:
                a1,b1=B[g1][:2]; a2,b2=B[g2][:2]
                if a2>=b1: break
                ov=min(b1,b2)-max(a1,a2)
                if ov>=0.3*min(b1-a1,b2-a2) and f(g1)!=f(g2): par[f(g2)]=f(g1); changed=True
        groups={}
        for i in keep: groups.setdefault(f(i),[]).append(i)
        B={g:box(v) for g,v in groups.items()}
        for g,(x0,x1,y0,y1,ar) in B.items():
            if max(x1-x0,y1-y0)>=0.25*hb: continue
            best=None
            for g2,(u0,u1,v0,v1,_) in B.items():
                if f(g2)==f(g): continue
                dx=max(0,u0-x1,x0-u1); dy=max(0,v0-y1,y0-v1); d=math.hypot(dx,dy)
                if d<0.15*hb and (best is None or d<best[0]): best=(d,g2)
            if best is not None: par[f(g)]=f(best[1]); changed=True
        if not changed: break
    groups={}
    for i in keep: groups.setdefault(f(i),[]).append(i)
    units=[]
    for g,v in groups.items():
        x0,x1,y0,y1,_=box(v); m=np.isin(lab,v); units.append(dict(x0=int(x0),x1=int(x1),y0=int(y0),y1=int(y1),mask=m))
    units.sort(key=lambda u:(u['x0'],u['x1']))
    return units,ink.astype(bool),drop.astype(bool),hb
def guard(px,pen,dist,thr,k):
    """pen-down runs of >= k points farther than thr px from the island ink: pen up before the run, run dropped.
    -> (kept index array, new pen)."""
    H,W=dist.shape; xi=np.clip(np.round(px[:,0]).astype(int),0,W-1); yi=np.clip(np.round(px[:,1]).astype(int),0,H-1)
    inside=(px[:,0]>=0)&(px[:,0]<W)&(px[:,1]>=0)&(px[:,1]<H); off=~inside|(dist[yi,xi]>thr)
    keep=np.ones(len(px),bool); newpen=pen.copy()
    if k<=0: return np.arange(len(px)),newpen
    for a,b in strokes_of(pen):
        if b-a+1<k: continue
        t=a
        while t<=b:
            if off[t]:
                u=t
                while u+1<=b and off[u+1]: u+=1
                if u-t+1>=k:
                    keep[t:u+1]=False
                    if t>a: newpen[t-1]=1
                    if u==b and t>a: newpen[t-1]=1
                t=u+1
            else: t+=1
    idx=np.where(keep)[0]; p2=newpen[idx]
    if len(p2): p2[-1]=1
    return idx,p2
def merge_strokes(px,pen,uid,w,k=1.5,cmin=0.5,cross=False):
    """a stroke torn by a break in the ink -> one stroke. Consecutive strokes A,B of the SAME island are glued (pen stays down over
    the jump) when |B0 - A_end| < k*w and the directions agree: cos(tangent at A's end, tangent at B's start) > cmin and, unless the
    jump is shorter than 0.5 w, cos(A's end tangent, jump) > cmin. Tangents over ~1.5 w of path. -> (new pen, number of glued joints)."""
    pen=pen.copy(); ab=strokes_of(pen); n=0
    def tang(q,end):
        if len(q)<2: return None
        seg=np.hypot(*np.diff(q,axis=0).T); c=np.concatenate([[0],np.cumsum(seg)]); L=c[-1]
        if L<1e-6: return None
        if end: j=int(np.searchsorted(c,max(0.0,L-1.5*w))); v=q[-1]-q[min(j,len(q)-2)]
        else: j=int(np.searchsorted(c,min(L,1.5*w))); v=q[min(max(j,1),len(q)-1)]-q[0]
        nv=float(np.hypot(*v)); return v/nv if nv>1e-6 else None
    for (a1,b1),(a2,b2) in zip(ab[:-1],ab[1:]):
        if uid[b1]!=uid[a2] and not cross: continue
        jump=px[a2]-px[b1]; d=float(np.hypot(*jump))
        if d>=k*w: continue
        t1=tang(px[a1:b1+1],True); t2=tang(px[a2:b2+1],False)
        if t1 is None or t2 is None or float(t1@t2)<=cmin: continue
        if d>=0.5*w and float(t1@(jump/d))<=cmin: continue
        pen[b1]=0; n+=1
    pen[-1]=1
    return pen,n

# ---------------------------------------------------------------- own line / neighbour ink
def _core(G):
    H = G.shape[0]; prof = G.sum(1).astype(float)
    a0, b0 = H // 4, 3 * H // 4; m = a0 + int(np.argmax(prof[a0:b0])); thr = CORE_FRAC * max(prof[m], 1.0); a = b = m
    while a > 0 and prof[a - 1] >= thr: a -= 1
    while b < H - 1 and prof[b + 1] >= thr: b += 1
    return a, b
def own_mask(x, derule=False):
    """derule rule-remnant pixels (rule_px) are taken out BEFORE the core band and the components — an underline is the densest
    row (it pulled the band onto itself) and it fuses the words it crosses with the neighbour lines' ink."""
    import cv2
    G = norm_img(x) > 0.5; H, W = G.shape; a, b = _core(G)
    if derule: G = G & ~rule_px(G, (a, b)); a, b = _core(G)
    n, lab, st, _ = cv2.connectedComponentsWithStats(G.astype(np.uint8), connectivity=8)
    inb = np.bincount(lab[a:b + 1].ravel(), minlength=n); tot = st[:, cv2.CC_STAT_AREA]
    keep = (inb >= CORE_MIN * tot) & (st[:, cv2.CC_STAT_HEIGHT] >= max(3, FLAT_H * (b - a + 1))); keep[0] = False
    return keep[lab], (a, b), lab, keep
def rule_px(G, core):
    """01.11: pixels of thin near-horizontal runs (rule remnants, often fused with the letters they cross): opening with thin line kernels
    of length max(24, 2.2 x core height) at -4..4 deg, minus parts thicker than 0.35 x core height (bold bars), dilated 1 px vertically."""
    import cv2
    hc = core[1] - core[0] + 1; L = max(24, int(2.2 * hc)); g = G.astype(np.uint8); oh = np.zeros_like(g)
    for ang in (-4, -2, 0, 2, 4):
        dy = int(round(np.tan(np.radians(ang)) * L / 2)); k = np.zeros((2 * abs(dy) + 1, L), np.uint8); cv2.line(k, (0, abs(dy) - dy), (L - 1, abs(dy) + dy), 1, 1)
        oh |= cv2.morphologyEx(g, cv2.MORPH_OPEN, k)
    thick = cv2.morphologyEx(oh, cv2.MORPH_OPEN, np.ones((max(3, int(0.35 * hc)), 1), np.uint8))
    return cv2.dilate((oh & (1 - thick)), np.ones((3, 1), np.uint8)) > 0
def neighbour_mask(G, own, core, lab, rule=None):
    """Ink of the NEIGHBOUR lines = not-own components (rule pixels aside) that touch the
    top/bottom frame edge (a line cut by the crop) or whose centre lies more than NB_FAR core heights beyond the core band; small marks
    near the band (commas, dots, accents, dashes) stay as they are (neither own nor neighbour)."""
    import cv2
    H, W = G.shape; a, b = core; hc = b - a + 1; oth = G & ~own & (~rule if rule is not None else True)
    n, lb, st, cen = cv2.connectedComponentsWithStats(oth.astype(np.uint8), connectivity=8); nb = np.zeros(n, bool)
    for c in range(1, n):
        y0 = st[c, cv2.CC_STAT_TOP]; y1 = y0 + st[c, cv2.CC_STAT_HEIGHT] - 1; cy = cen[c, 1]
        nb[c] = y0 <= 0 or y1 >= H - 1 or cy < a - NB_FAR * hc or cy > b + NB_FAR * hc
    return nb[lb]

# ---------------------------------------------------------------- the line tracer
@torch.no_grad()
def restore(model, L, bs=16):
    """uint8 line frame (height 64) -> restored uint8 frame (paper 255, ink 0 = 255*(1-p))."""
    dev = next(model.parameters()).device; Wm = int(math.ceil(L.shape[1] / 8) * 8); x = np.zeros((1, 1, L.shape[0], Wm), np.float32); x[0, 0, :, :L.shape[1]] = norm_img(L)
    p = torch.sigmoid(model(torch.from_numpy(x).to(dev)).float()).cpu().numpy()
    return np.clip(np.round(255 * (1 - p[0, 0, :, :L.shape[1]])), 0, 255).astype(np.uint8)
def _erase_neighbours(L, paper):
    G_ = norm_img(L) > 0.5; o_, c_, lb_, _k = own_mask(L, derule=True); r_ = rule_px(G_, c_) & G_; o_ = o_ & ~r_; nb_ = neighbour_mask(G_, o_, c_, lb_, r_)
    nb_ = cv2.dilate(nb_.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0; nb_ &= ~cv2.dilate(o_.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool); L = L.copy(); L[nb_] = paper
    return L
def trace_frame(model, L, residual_pass=False, erase_neighbours=False):
    """L restored line frame (height 64, any width) -> dict(px (n,2) frame px, pen, pres, uid, w)."""
    dv = next(model.parameters()).device; L, _ = clean_rules(L); imh = model.cfg['IMH']; H, W = L.shape; paper = int(np.median(L))
    if erase_neighbours and not residual_pass: L = _erase_neighbours(L, paper)
    G0 = norm_img(L) > 0.5; wb = float(np.clip(G0.sum() / max(skel_len(skeletonize(G0)), 1.0), 1.0, 8.0)); units, ink, drop, hb = find_islands(L, BRIDGE * wb)
    units = split_long_units(units, ink, SPLIT_LONG)
    sk = skeletonize(ink); w = float(np.clip(ink.sum() / max(skel_len(sk), 1.0), 1.0, 8.0))
    if not units: return dict(px=np.zeros((1, 2), np.float32), pen=np.ones(1), pres=np.full(1, 0.5, np.float32), uid=np.zeros(1, int), w=w)
    allm = np.zeros((H, W), np.uint8)
    for u in units: allm |= u['mask'].astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)); dropd = cv2.dilate(drop.astype(np.uint8), k3).astype(bool)
    m = int(round(0.25 * imh)); wins = []
    for u in units:
        a = max(0, u['x0'] - m); b = min(W, u['x1'] + m)
        if b - a < imh // 2: c = (a + b) // 2; a = max(0, c - imh // 4); b = min(W, a + imh // 2); a = max(0, b - imh // 2)
        own = u['mask'].astype(np.uint8); other = cv2.dilate((allm & (1 - own)), k3).astype(bool) | dropd
        other &= ~cv2.dilate(own, np.ones((3, 3), np.uint8)).astype(bool)
        crop = L.copy(); crop[other] = paper; wins.append((a, b, crop[:, a:b]))
    order = np.argsort([b - a for a, b, _ in wins]); res = [None] * len(wins)
    for s in range(0, len(order), IBS):
        ids = order[s:s + IBS]; Wm = int(math.ceil(max(wins[i][1] - wins[i][0] for i in ids) / 8) * 8)
        img = np.zeros((len(ids), 1, imh, Wm), np.float32); wc = np.zeros(len(ids), np.int64)
        for q, i in enumerate(ids): x_ = wins[i][2]; img[q, 0, :, :x_.shape[1]] = norm_img(x_); wc[q] = math.ceil(x_.shape[1] / 8)
        ml = int(min(MAXPTS, TRACE_ML_K * Wm + 100))
        trs = trace(model, torch.from_numpy(img).to(dv), torch.from_numpy(wc).to(dv), bias=SAMPLE_BIAS, maxlen=ml)
        for i, tr in zip(ids, trs): res[i] = tr
    P = []; Pn = []; Pr = []; U = []
    for u, (a, b, _), tr in zip(units, wins, res):
        px = tr['absu'] * imh + np.array([a, 0], np.float32); pen = tr['pen'].copy(); pen[-1] = 1
        dist = cv2.distanceTransform((1 - u['mask'].astype(np.uint8)), cv2.DIST_L2, 3)
        idx, p2 = guard(px, pen, dist, GUARD_W * w, GUARD_K)
        if len(idx):   # strokes with every point off this island's ink are dropped
            q = px[idx]; xi_ = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1); yi_ = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1); far = dist[yi_, xi_] > GUARD_W * w
            kk = np.ones(len(idx), bool)
            for a_, b_ in strokes_of(p2):
                if far[a_:b_ + 1].all(): kk[a_:b_ + 1] = False
            idx = idx[kk]; p2 = p2[kk]
            if len(p2): p2[-1] = 1
        if not len(idx): continue
        P.append(px[idx]); Pn.append(p2); Pr.append(tr['pres'][idx] if tr['pres'] is not None else np.full(len(idx), 0.5, np.float32)); U.append(np.full(len(idx), len(U)))
    if not P: P = [np.zeros((1, 2), np.float32)]; Pn = [np.ones(1)]; Pr = [np.full(1, 0.5, np.float32)]; U = [np.zeros(1, int)]
    Pc, Pnc, Prc, Uc = np.concatenate(P), np.concatenate(Pn), np.concatenate(Pr), np.concatenate(U)
    if len(Pc) > 2 and not residual_pass:   # second pass: own ink of the kept units left untraced
        cov = np.zeros((H, W), np.uint8)
        for a_, b_ in strokes_of(Pnc):
            q = np.round(Pc[a_:b_ + 1]).astype(np.int32)
            if len(q) > 1: cv2.polylines(cov, [q.reshape(-1, 1, 2)], False, 1, thickness=max(1, int(round(2 * RES_R * w))))
            else: cv2.circle(cov, tuple(int(v) for v in q[0]), max(1, int(round(RES_R * w))), 1, -1)
        res_m = (allm > 0) & (cov == 0); res_m &= own_mask(L)[0]
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(res_m.astype(np.uint8), 8)
        big = [i for i in range(1, n_) if st_[i, cv2.CC_STAT_AREA] >= RES_MIN]
        if big:
            Lr = np.full_like(L, paper); keep_m = np.isin(lab_, big); Lr[keep_m] = L[keep_m]
            r2 = trace_frame(model, Lr, residual_pass=True)
            if len(r2['px']) > 1:
                Pc = np.concatenate([Pc, r2['px']]); Pnc = np.concatenate([Pnc, r2['pen']]); Prc = np.concatenate([Prc, r2['pres']]); Uc = np.concatenate([Uc, r2['uid'] + Uc.max() + 1])
    return dict(px=Pc, pen=Pnc, pres=Prc, uid=Uc, w=w)
class Tracer:
    def __init__(s, weights='weights', device=None):
        s.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
        st = torch.load(os.path.join(weights, 'trace8_fp16.pt'), map_location='cpu', weights_only=True); s.model = Model(st['cfg']); s.model.load_state_dict({k: v.float() for k, v in st['e'].items()}); s.model.to(s.dev).eval()
        st = torch.load(os.path.join(weights, 'inkfix1.pt'), map_location='cpu', weights_only=True); s.fix = UNet(**st['cfg']); s.fix.load_state_dict({k: v.float() for k, v in st['e'].items()}); s.fix.to(s.dev).eval()
    def trace_frame64(s, frame, seed=0, erase_neighbours=False):
        """uint8 frame of height 64 -> trace in frame px, stroke merge included."""
        torch.manual_seed(seed); L = restore(s.fix, frame); r = trace_frame(s.model, L, erase_neighbours=erase_neighbours)
        pen, _ = merge_strokes(r['px'], r['pen'], r['uid'], r['w'], MERGE, MERGE_COS, True); return dict(px=np.asarray(r['px'], np.float32), pen=np.asarray(pen, np.float32), pres=np.asarray(r['pres'], np.float32))
    def trace_image(s, gray, seed=0, erase_neighbours=False):
        """uint8 grey crop of one line, any size -> trace in the crop's pixels."""
        f = to_input(gray, s.model.cfg['IMH'], 10 ** 9); r = s.trace_frame64(f, seed, erase_neighbours); r['px'] = r['px'] * (gray.shape[0] / s.model.cfg['IMH']); return r
def to_svg(px, pen, w, h):
    d = []; a = 0
    for k in range(len(px)):
        if pen[k] > 0.5 or k == len(px) - 1:
            d.append('M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in px[a:k + 1])); a = k + 1
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path d="{" ".join(d)}" fill="none" stroke="black" stroke-width="{max(1.0, h / 40):.1f}" stroke-linecap="round" stroke-linejoin="round"/></svg>'
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('image'); ap.add_argument('--weights', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')); ap.add_argument('--out', default='trace.json')
    ap.add_argument('--svg'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device'); ap.add_argument('--erase_neighbours', action='store_true'); A = ap.parse_args()
    g = np.asarray(Image.open(A.image).convert('L')); r = Tracer(A.weights, A.device).trace_image(g, A.seed, A.erase_neighbours)
    json.dump(dict(height=g.shape[0], width=g.shape[1], points=[[round(float(x), 2), round(float(y), 2), int(p > 0.5), round(float(q), 3)] for (x, y), p, q in zip(r['px'], r['pen'], r['pres'])]), open(A.out, 'w'))
    if A.svg: open(A.svg, 'w').write(to_svg(r['px'], r['pen'], g.shape[1], g.shape[0]))
    print(A.out, len(r['px']), 'points', int((r['pen'] > 0.5).sum()), 'strokes')
