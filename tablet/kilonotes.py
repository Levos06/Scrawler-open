"""Kilonotes zip exports (Android) -> pen strokes of every page.

The file is only read as data. The format has no public description; these notes are reverse-engineered from one export.

Layout: /<doc-uuid>/info (JSON: title, width and height in points, pages = ordered page uuids) and
/<doc-uuid>/pages/<page-uuid>/info (JSON: draws[], where '_ct': 'ct_pen' is a handwriting stroke and 'ct_stroke' another drawn
object; points [{x, y, pressure, eventTime in ms, discontinuity}]; matrix = 3x3 row-major [a b c; d e f; 0 0 1] that maps
stroke-local points to page coordinates; strokeWidth, strokeColor, visible). Only visible ct_pen strokes are read, ordered by
the time of their first point.
"""
import json
import zipfile


def read(path):
    """-> the pages in notebook order: [{"width", "height", "strokes": [[[x, y], ...], ...]}], strokes in writing order."""
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    out = []
    for root in sorted(n for n in names if n.endswith('/info') and n.strip('/').count('/') == 1):
        base = root[: -len('info')]
        info = json.loads(z.read(root))
        for page in info.get('pages', []):
            name = f"{base}pages/{page if isinstance(page, str) else page.get('uuid')}/info"
            if name not in names:
                continue
            d = json.loads(z.read(name))
            paper = d.get('paper', {})
            strokes = []
            for q in d.get('draws', []):
                points = q.get('points') or []
                if not q.get('visible', True) or q.get('_ct') != 'ct_pen' or not points:
                    continue
                a, b, c, dd, e, f = (q.get('matrix') or [1, 0, 0, 0, 1, 0, 0, 0, 1])[:6]
                t0 = points[0].get('eventTime')
                strokes.append((t0 is None, t0 or 0, len(strokes), [[a * p['x'] + b * p['y'] + c, dd * p['x'] + e * p['y'] + f] for p in points]))
            strokes.sort(key=lambda s: s[:3])
            out.append(dict(width=paper.get('widthInPoint') or info.get('width'), height=paper.get('heightInPoint') or info.get('height'), strokes=[s[3] for s in strokes]))
    return out
