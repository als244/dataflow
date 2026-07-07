
### STATIC placement

| dev GiB | dsv3 (dense MLA) | dsv32 (DSA) | glm52 (DSA+IndexShare) |
|---|---|---|---|
| 12 | 5,095 (sim 3,879) · 10.81 GiB · bs4ga4 · rc 100% | 5,124 (sim 4,103) · 11.92 GiB · bs4ga4 · rc 100% | 3,288 (sim 3,691) · 11.81 GiB · bs4ga4 · rc 100% |
| 16 | 8,204 (sim 7,239) · 15.80 GiB · bs8ga2 · rc 81% | 7,618 (sim 6,742) · 15.90 GiB · bs8ga2 · rc 75% | 7,457 (sim 6,546) · 15.70 GiB · bs8ga2 · rc 100% |
| 20 | 8,582 (sim 7,750) · 20.00 GiB · bs8ga2 · rc 75% | 8,175 (sim 7,108) · 19.61 GiB · bs8ga2 · rc 69% | 8,213 (sim 7,214) · 19.75 GiB · bs8ga2 · rc 56% |
| 24 | 10,335 (sim 9,089) · 22.62 GiB · bs16ga1 · rc 89% | 9,053 (sim 7,670) · 22.23 GiB · bs16ga1 · rc 100% | 9,391 (sim 7,752) · 23.16 GiB · bs16ga1 · rc 100% |
| 28 | 12,177 (sim 12,083) · 28.00 GiB · bs16ga1 · rc 83% | 10,427 (sim 9,155) · 27.47 GiB · bs16ga1 · rc 78% | 11,270 (sim 10,103) · 27.98 GiB · bs16ga1 · rc 72% |

### VMM placement

| dev GiB | dsv3 (dense MLA) | dsv32 (DSA) | glm52 (DSA+IndexShare) |
|---|---|---|---|
| 12 | 5,174 (sim 4,159) · 11.19 GiB · bs4ga4 · rc 100% | 4,694 (sim 3,871) · 11.19 GiB · bs4ga4 · rc 100% | 5,286 (sim 4,512) · 11.95 GiB · bs4ga4 · rc 79% |
| 16 | 7,913 (sim 7,289) · 15.71 GiB · bs8ga2 · rc 100% | 7,563 (sim 6,759) · 15.31 GiB · bs8ga2 · rc 75% | 7,991 (sim 7,146) · 15.31 GiB · bs8ga2 · rc 61% |
| 20 | 8,289 (sim 7,806) · 19.75 GiB · bs8ga2 · rc 75% | 7,998 (sim 7,219) · 19.71 GiB · bs8ga2 · rc 61% | 8,220 (sim 7,547) · 19.37 GiB · bs8ga2 · rc 69% |
| 24 | 11,988 (sim 12,192) · 23.93 GiB · bs16ga1 · rc 89% | 9,931 (sim 9,095) · 23.56 GiB · bs16ga1 · rc 78% | 9,856 (sim 8,374) · 23.56 GiB · bs16ga1 · rc 72% |
| 28 | 11,963 (sim 11,900) · 27.99 GiB · bs16ga1 · rc 78% | 10,333 (sim 9,815) · 27.93 GiB · bs16ga1 · rc 67% | 10,954 (sim 9,834) · 27.67 GiB · bs16ga1 · rc 83% |

### Best legal per cell (mode in parens where not static)

| dev GiB | dsv3 (dense MLA) | dsv32 (DSA) | glm52 (DSA+IndexShare) |
|---|---|---|---|
| 12 | 5,174 (vmm) | 5,124 | 5,286 (vmm) |
| 16 | 8,204 | 7,618 | 7,991 (vmm) |
| 20 | 8,582 | 8,175 | 8,220 (vmm) |
| 24 | 11,988 (vmm) | 9,931 (vmm) | 9,856 (vmm) |
| 28 | 12,177 | 10,427 | 11,270 |
