# Tracer benchmark, part A: true pen, lines unseen in training

F1 at R = 2 px on a 64 px frame (recall in brackets). Precision is in the JSON reports kept locally. 2026-11-06.

| source, lines | look | previous recipe | released | with --erase_neighbours |
|---|---|---|---|---|
| all, 2677 | clean | 0.954 (0.935) | **0.960 (0.945)** | 0.941 (0.918) |
| all, 2677 | scan | 0.938 (0.915) | **0.945 (0.938)** | 0.931 (0.909) |
| all, 2677 | photo | 0.935 (0.912) | **0.943 (0.938)** | 0.933 (0.914) |
| English (BRUSH), 600 | clean | 0.981 (0.979) | **0.984 (0.984)** | 0.983 (0.982) |
| English (BRUSH), 600 | scan | 0.974 (0.968) | **0.974 (0.978)** | 0.976 (0.974) |
| English (BRUSH), 600 | photo | 0.971 (0.967) | **0.972 (0.977)** | 0.975 (0.975) |
| Russian tablet prose, 830 | clean | 0.952 (0.922) | **0.955 (0.926)** | 0.943 (0.908) |
| Russian tablet prose, 830 | scan | 0.936 (0.903) | **0.941 (0.921)** | 0.934 (0.904) |
| Russian tablet prose, 830 | photo | 0.934 (0.900) | **0.941 (0.920)** | 0.935 (0.905) |
| Russian tablet math, 300 | clean | 0.961 (0.941) | **0.962 (0.945)** | 0.922 (0.886) |
| Russian tablet math, 300 | scan | 0.938 (0.912) | **0.947 (0.941)** | 0.920 (0.890) |
| Russian tablet math, 300 | photo | 0.941 (0.921) | **0.942 (0.943)** | 0.926 (0.904) |
| MathWriting, 500 | clean | 0.939 (0.915) | **0.945 (0.926)** | 0.891 (0.848) |
| MathWriting, 500 | scan | 0.910 (0.878) | **0.921 (0.911)** | 0.869 (0.825) |
| MathWriting, 500 | photo | 0.907 (0.873) | **0.922 (0.918)** | 0.879 (0.842) |
| Korean, 297 | clean | 0.937 (0.915) | **0.956 (0.947)** | 0.956 (0.947) |
| Korean, 297 | scan | 0.924 (0.897) | **0.939 (0.934)** | 0.940 (0.931) |
| Korean, 297 | photo | 0.920 (0.893) | **0.935 (0.933)** | 0.938 (0.932) |
| Urdu (OUHD-L), 150 | clean | 0.925 (0.928) | **0.942 (0.959)** | 0.941 (0.959) |
| Urdu (OUHD-L), 150 | scan | 0.927 (0.930) | **0.936 (0.955)** | 0.941 (0.954) |
| Urdu (OUHD-L), 150 | photo | 0.914 (0.909) | **0.929 (0.949)** | 0.934 (0.944) |
