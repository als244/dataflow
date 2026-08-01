# FA3 corrected vs FA3 original — what sustained-load profiling changed

Same grid, same box (della H100 80GB, llama3_8b, 128K tokens/step), same
FA3 kernels. **Both campaigns planned the same way** — measured task profiles
plus this box's own measured PCIe. What changed is HOW those task profiles were
taken: the rerun samples every signature under SUSTAINED load (a 2.5 s floor
per signature instead of a short burst) and seeds weights with the production
initialisation. Nothing here is a roofline-vs-measured comparison, and the
sim-vs-real planning gap was not in play for either run.
84 measured frontier cells per optimizer; 69 share a
(seq, tokens/round, tokens/step, budget) key with the original and are the
basis for every cell-by-cell number here.

Sources: `results_h100_fa3_corrected/` (this run) and `results_h100_fa3/`
(original). Both carry `data/measure_{adamw,muon}.jsonl`.

## Headline

Throughput is over the 69 cells common to both campaigns; fidelity is over all
84 measured cells in each.

| | adamw | muon |
|---|---|---|
| measured throughput, median | **-2.1%** | **+88.7%** |
| effective TFLOP/s, median | 512 -> 503 | **268 -> 507** |
| ratio mean | 1.036 -> **1.024** | 1.029 -> 1.008 |
| ratio range | 1.01-1.10 -> **1.01-1.04** | 1.02-1.07 -> **0.87-1.07** |
| worst absolute error | 10% -> **4%** | 7% -> **13%** |
| cells off by >5% | 10 -> **0** | 8 -> 1 |

Read these as three separate results, not one.

1. **adamw fidelity improved cleanly.** Worst-case error 10% -> 4%, and the
   ten cells that were off by more than 5% are all gone. Nothing else in this
   report qualifies that.
2. **Muon throughput nearly doubled** (median +88.7%, effective TFLOP/s
   268 -> 507). Muon now runs at adamw's effective TFLOP/s on the same box,
   which it did not before; what in the revision delta produced that is not
   established here.
3. **Muon fidelity got better on average and worse at the edge.** The mean
   moved to 1.008, but only because a new population of cells now measures
   FASTER than predicted; the worst absolute error went from 7% to 13%. It is
   entirely seq 4096, and it is a new problem.

## adamw

### Fidelity (measured / predicted; 1.00 is perfect)

Restricted to the 69 common cells, so it differs slightly from the
headline, which uses all 84.

| | n | mean | median | min | max | >1.05 | <1.00 |
|---|---|---|---|---|---|---|---|
| original | 69 | 1.032 | 1.028 | 1.01 | 1.10 | 3 | 0 |
| corrected | 69 | 1.025 | 1.028 | 1.01 | 1.04 | 0 | 0 |

### Measured throughput, cell by cell

Median **-2.1%**, mean -1.9%, range -3.7% to +1.8%. 1 cells faster, 66 slower, 2 unchanged.

| sequence | cells | median change |
|---|---|---|
| 1024 | 17 | -1.2% |
| 2048 | 16 | -1.5% |
| 4096 | 20 | -2.2% |
| 8192 | 16 | -2.3% |

Plans are not the same plans: **0 of 69 cells kept the same `prog_id`**, and the planner now recomputes more (median +6 layers, more in 43 cells, fewer in 3). Sustained-load profiling moves the task prices, and the planner searches from different prices, so it lands on different programs. Which way the recompute count moves is an observation here, not something this report explains.

<details><summary>all 69 common cells</summary>

| seq | t/round | t/step | budget | old tok/s | new tok/s | change | old ratio | new ratio |
|---|---|---|---|---|---|---|---|---|
| 1024 | 32K | 64K | 8.8 | 10,302 | 10,209 | -0.9% | 1.04 | 1.03 |
| 1024 | 32K | 128K | 8.8 | 10,413 | 10,302 | -1.1% | 1.04 | 1.04 |
| 1024 | 32K | 256K | 8.8 | 10,477 | 10,355 | -1.2% | 1.04 | 1.04 |
| 1024 | 16K | 64K | 12 | 10,278 | 10,461 | +1.8% | 1.04 | 1.02 |
| 1024 | 32K | 128K | 12 | 10,403 | 10,299 | -1.0% | 1.04 | 1.04 |
| 1024 | 32K | 64K | 18 | 11,162 | 10,960 | -1.8% | 1.02 | 1.03 |
| 1024 | 32K | 128K | 18 | 11,026 | 10,659 | -3.3% | 1.02 | 1.04 |
| 1024 | 32K | 256K | 18 | 11,002 | 10,805 | -1.8% | 1.02 | 1.01 |
| 1024 | 32K | 64K | 25 | 11,351 | 11,258 | -0.8% | 1.04 | 1.03 |
| 1024 | 32K | 128K | 25 | 11,165 | 11,038 | -1.1% | 1.03 | 1.02 |
| 1024 | 16K | 64K | 35 | 11,687 | 11,542 | -1.2% | 1.02 | 1.02 |
| 1024 | 16K | 128K | 35 | 11,752 | 11,566 | -1.6% | 1.03 | 1.02 |
| 1024 | 16K | 64K | 50 | 12,229 | 12,006 | -1.8% | 1.02 | 1.01 |
| 1024 | 16K | 128K | 50 | 12,380 | 12,022 | -2.9% | 1.03 | 1.02 |
| 1024 | 16K | 256K | 50 | 12,552 | 12,294 | -2.1% | 1.03 | 1.02 |
| 1024 | 16K | 128K | 67.3 | 12,494 | 12,367 | -1.0% | 1.02 | 1.02 |
| 1024 | 16K | 256K | 67.3 | 12,618 | 12,511 | -0.8% | 1.02 | 1.02 |
| 2048 | 32K | 64K | 8.8 | 10,119 | 10,007 | -1.1% | 1.03 | 1.03 |
| 2048 | 32K | 128K | 8.8 | 10,233 | 10,113 | -1.2% | 1.03 | 1.03 |
| 2048 | 32K | 256K | 8.8 | 10,263 | 10,133 | -1.3% | 1.03 | 1.03 |
| 2048 | 32K | 128K | 12 | 10,241 | 10,123 | -1.2% | 1.03 | 1.03 |
| 2048 | 32K | 256K | 12 | 10,724 | 10,425 | -2.8% | 1.01 | 1.03 |
| 2048 | 32K | 64K | 18 | 10,997 | 10,830 | -1.5% | 1.02 | 1.03 |
| 2048 | 32K | 128K | 18 | 10,803 | 10,521 | -2.6% | 1.01 | 1.03 |
| 2048 | 32K | 256K | 18 | 10,783 | 10,483 | -2.8% | 1.01 | 1.03 |
| 2048 | 32K | 64K | 25 | 11,143 | 11,036 | -1.0% | 1.03 | 1.03 |
| 2048 | 32K | 128K | 25 | 10,924 | 10,873 | -0.5% | 1.03 | 1.02 |
| 2048 | 8K | 256K | 35 | 11,464 | 11,293 | -1.5% | 1.08 | 1.03 |
| 2048 | 16K | 64K | 35 | 11,496 | 11,359 | -1.2% | 1.02 | 1.02 |
| 2048 | 16K | 64K | 50 | 11,928 | 11,682 | -2.1% | 1.02 | 1.02 |
| 2048 | 16K | 128K | 50 | 12,207 | 11,807 | -3.3% | 1.02 | 1.02 |
| 2048 | 16K | 256K | 50 | 12,285 | 12,026 | -2.1% | 1.02 | 1.02 |
| 2048 | 32K | 64K | 67.3 | 12,138 | 11,963 | -1.4% | 1.04 | 1.03 |
| 4096 | 32K | 64K | 8.8 | 9,751 | 9,798 | +0.5% | 1.04 | 1.03 |
| 4096 | 32K | 128K | 8.8 | 10,233 | 9,983 | -2.4% | 1.01 | 1.02 |
| 4096 | 32K | 256K | 8.8 | 10,296 | 9,930 | -3.6% | 1.01 | 1.03 |
| 4096 | 32K | 64K | 12 | 10,202 | 10,084 | -1.2% | 1.03 | 1.02 |
| 4096 | 32K | 128K | 12 | 10,344 | 9,958 | -3.7% | 1.02 | 1.03 |
| 4096 | 32K | 256K | 12 | 10,404 | 10,118 | -2.8% | 1.02 | 1.02 |
| 4096 | 32K | 64K | 18 | 10,712 | 10,410 | -2.8% | 1.02 | 1.03 |
| 4096 | 32K | 128K | 18 | 10,444 | 10,155 | -2.8% | 1.02 | 1.03 |
| 4096 | 32K | 256K | 18 | 10,459 | 10,175 | -2.7% | 1.02 | 1.03 |
| 4096 | 16K | 256K | 25 | 10,618 | 10,368 | -2.4% | 1.04 | 1.03 |
| 4096 | 32K | 64K | 25 | 10,726 | 10,504 | -2.1% | 1.04 | 1.03 |
| 4096 | 32K | 128K | 25 | 10,643 | 10,480 | -1.5% | 1.03 | 1.02 |
| 4096 | 8K | 256K | 35 | 11,039 | 10,808 | -2.1% | 1.08 | 1.04 |
| 4096 | 16K | 64K | 35 | 11,132 | 10,896 | -2.1% | 1.04 | 1.01 |
| 4096 | 16K | 128K | 35 | 11,220 | 10,915 | -2.7% | 1.04 | 1.02 |
| 4096 | 16K | 64K | 50 | 11,533 | 11,319 | -1.9% | 1.03 | 1.01 |
| 4096 | 16K | 128K | 50 | 11,780 | 11,510 | -2.3% | 1.03 | 1.02 |
| 4096 | 16K | 256K | 50 | 11,838 | 11,582 | -2.2% | 1.04 | 1.02 |
| 4096 | 16K | 256K | 67.3 | 11,935 | 11,708 | -1.9% | 1.04 | 1.02 |
| 4096 | 32K | 64K | 67.3 | 11,800 | 11,542 | -2.2% | 1.04 | 1.03 |
| 8192 | 32K | 128K | 8.8 | 9,173 | 8,962 | -2.3% | 1.05 | 1.03 |
| 8192 | 32K | 256K | 8.8 | 9,645 | 9,330 | -3.3% | 1.02 | 1.02 |
| 8192 | 32K | 64K | 12 | 9,447 | 9,309 | -1.5% | 1.01 | 1.03 |
| 8192 | 32K | 128K | 12 | 9,706 | 9,378 | -3.4% | 1.02 | 1.03 |
| 8192 | 32K | 256K | 12 | 9,834 | 9,561 | -2.8% | 1.02 | 1.02 |
| 8192 | 32K | 64K | 18 | 9,952 | 9,765 | -1.9% | 1.05 | 1.03 |
| 8192 | 32K | 128K | 18 | 9,781 | 9,447 | -3.4% | 1.03 | 1.03 |
| 8192 | 32K | 256K | 18 | 9,869 | 9,646 | -2.3% | 1.03 | 1.01 |
| 8192 | 16K | 256K | 25 | 9,949 | 9,744 | -2.1% | 1.05 | 1.01 |
| 8192 | 32K | 64K | 25 | 10,014 | 9,767 | -2.5% | 1.05 | 1.03 |
| 8192 | 32K | 128K | 25 | 9,983 | 9,747 | -2.4% | 1.04 | 1.02 |
| 8192 | 8K | 256K | 35 | 10,402 | 10,167 | -2.3% | 1.10 | 1.03 |
| 8192 | 16K | 64K | 50 | 10,582 | 10,440 | -1.3% | 1.04 | 1.01 |
| 8192 | 16K | 128K | 50 | 10,802 | 10,659 | -1.3% | 1.05 | 1.01 |
| 8192 | 32K | 64K | 67.3 | 11,069 | 10,824 | -2.2% | 1.05 | 1.03 |
| 8192 | 32K | 128K | 67.3 | 11,209 | 10,843 | -3.3% | 1.03 | 1.03 |

</details>

## muon

### Fidelity (measured / predicted; 1.00 is perfect)

Restricted to the 69 common cells, so it differs slightly from the
headline, which uses all 84.

| | n | mean | median | min | max | >1.05 | <1.00 |
|---|---|---|---|---|---|---|---|
| original | 69 | 1.027 | 1.024 | 1.02 | 1.07 | 3 | 0 |
| corrected | 69 | 1.005 | 1.025 | 0.87 | 1.07 | 1 | 20 |

### Measured throughput, cell by cell

Median **+88.7%**, mean +102.3%, range +40.2% to +187.6%. 69 cells faster, 0 slower, 0 unchanged.

| sequence | cells | median change |
|---|---|---|
| 1024 | 17 | +94.3% |
| 2048 | 16 | +89.8% |
| 4096 | 20 | +84.8% |
| 8192 | 16 | +80.1% |

Plans are not the same plans: **0 of 69 cells kept the same `prog_id`**, and the planner now recomputes more (median +0 layers, more in 28 cells, fewer in 8). Sustained-load profiling moves the task prices, and the planner searches from different prices, so it lands on different programs. Which way the recompute count moves is an observation here, not something this report explains.

<details><summary>all 69 common cells</summary>

| seq | t/round | t/step | budget | old tok/s | new tok/s | change | old ratio | new ratio |
|---|---|---|---|---|---|---|---|---|
| 1024 | 32K | 64K | 8.8 | 3,296 | 8,613 | +161.3% | 1.03 | 1.04 |
| 1024 | 32K | 128K | 8.8 | 5,021 | 9,427 | +87.7% | 1.03 | 1.04 |
| 1024 | 32K | 256K | 8.8 | 6,788 | 10,007 | +47.4% | 1.04 | 1.04 |
| 1024 | 16K | 64K | 12 | 3,354 | 8,896 | +165.2% | 1.02 | 1.01 |
| 1024 | 32K | 128K | 12 | 5,020 | 9,681 | +92.8% | 1.03 | 1.03 |
| 1024 | 32K | 64K | 18 | 3,381 | 9,277 | +174.4% | 1.02 | 1.02 |
| 1024 | 32K | 128K | 18 | 5,155 | 9,728 | +88.7% | 1.02 | 1.04 |
| 1024 | 32K | 256K | 18 | 6,998 | 10,275 | +46.8% | 1.02 | 1.02 |
| 1024 | 32K | 64K | 25 | 3,394 | 9,305 | +174.1% | 1.03 | 1.04 |
| 1024 | 32K | 128K | 25 | 5,180 | 10,066 | +94.3% | 1.03 | 1.02 |
| 1024 | 16K | 64K | 35 | 3,432 | 9,351 | +172.4% | 1.02 | 1.03 |
| 1024 | 16K | 128K | 35 | 5,264 | 10,126 | +92.4% | 1.02 | 1.03 |
| 1024 | 16K | 64K | 50 | 3,499 | 9,975 | +185.1% | 1.02 | 1.03 |
| 1024 | 16K | 128K | 50 | 5,479 | 11,095 | +102.5% | 1.02 | 1.03 |
| 1024 | 16K | 256K | 50 | 7,642 | 11,701 | +53.1% | 1.02 | 1.03 |
| 1024 | 16K | 128K | 67.3 | 5,487 | 11,149 | +103.2% | 1.02 | 1.02 |
| 1024 | 16K | 256K | 67.3 | 7,657 | 11,714 | +53.0% | 1.02 | 1.03 |
| 2048 | 32K | 64K | 8.8 | 3,311 | 8,618 | +160.3% | 1.02 | 1.04 |
| 2048 | 32K | 128K | 8.8 | 5,047 | 9,379 | +85.8% | 1.02 | 1.04 |
| 2048 | 32K | 256K | 8.8 | 6,847 | 9,879 | +44.3% | 1.02 | 1.03 |
| 2048 | 32K | 128K | 12 | 5,073 | 9,484 | +87.0% | 1.02 | 1.03 |
| 2048 | 32K | 256K | 12 | 6,900 | 9,890 | +43.3% | 1.02 | 1.04 |
| 2048 | 32K | 64K | 18 | 3,369 | 9,134 | +171.1% | 1.02 | 1.02 |
| 2048 | 32K | 128K | 18 | 5,118 | 9,583 | +87.2% | 1.02 | 1.03 |
| 2048 | 32K | 256K | 18 | 6,921 | 10,072 | +45.5% | 1.02 | 1.03 |
| 2048 | 32K | 64K | 25 | 3,376 | 9,164 | +171.5% | 1.03 | 1.04 |
| 2048 | 32K | 128K | 25 | 5,138 | 9,886 | +92.4% | 1.03 | 1.02 |
| 2048 | 8K | 256K | 35 | 7,334 | 10,983 | +49.7% | 1.06 | 1.04 |
| 2048 | 16K | 64K | 35 | 3,410 | 9,155 | +168.5% | 1.02 | 1.03 |
| 2048 | 16K | 64K | 50 | 3,482 | 9,813 | +181.9% | 1.02 | 1.03 |
| 2048 | 16K | 128K | 50 | 5,440 | 10,877 | +99.9% | 1.02 | 1.03 |
| 2048 | 16K | 256K | 50 | 7,559 | 11,435 | +51.3% | 1.02 | 1.03 |
| 2048 | 32K | 64K | 67.3 | 3,465 | 9,966 | +187.6% | 1.03 | 1.03 |
| 4096 | 32K | 64K | 8.8 | 3,278 | 8,369 | +155.3% | 1.02 | 0.89 |
| 4096 | 32K | 128K | 8.8 | 4,975 | 9,160 | +84.1% | 1.02 | 0.94 |
| 4096 | 32K | 256K | 8.8 | 6,722 | 9,705 | +44.4% | 1.02 | 0.97 |
| 4096 | 32K | 64K | 12 | 3,297 | 8,368 | +153.8% | 1.02 | 0.90 |
| 4096 | 32K | 128K | 12 | 4,999 | 9,111 | +82.3% | 1.02 | 0.95 |
| 4096 | 32K | 256K | 12 | 6,774 | 9,653 | +42.5% | 1.02 | 0.98 |
| 4096 | 32K | 64K | 18 | 3,335 | 8,433 | +152.9% | 1.02 | 0.92 |
| 4096 | 32K | 128K | 18 | 5,039 | 9,213 | +82.8% | 1.03 | 0.95 |
| 4096 | 32K | 256K | 18 | 6,807 | 9,639 | +41.6% | 1.02 | 0.99 |
| 4096 | 16K | 256K | 25 | 6,906 | 10,019 | +45.1% | 1.03 | 0.98 |
| 4096 | 32K | 64K | 25 | 3,337 | 8,496 | +154.6% | 1.03 | 0.92 |
| 4096 | 32K | 128K | 25 | 5,056 | 9,376 | +85.4% | 1.03 | 0.96 |
| 4096 | 8K | 256K | 35 | 7,203 | 10,590 | +47.0% | 1.06 | 0.99 |
| 4096 | 16K | 64K | 35 | 3,379 | 9,033 | +167.4% | 1.02 | 0.88 |
| 4096 | 16K | 128K | 35 | 5,128 | 9,892 | +92.9% | 1.03 | 0.94 |
| 4096 | 16K | 64K | 50 | 3,446 | 9,478 | +175.1% | 1.02 | 0.87 |
| 4096 | 16K | 128K | 50 | 5,349 | 10,490 | +96.1% | 1.03 | 0.93 |
| 4096 | 16K | 256K | 50 | 7,388 | 11,065 | +49.8% | 1.03 | 0.98 |
| 4096 | 16K | 256K | 67.3 | 7,402 | 11,091 | +49.8% | 1.03 | 0.97 |
| 4096 | 32K | 64K | 67.3 | 3,438 | 8,928 | +159.7% | 1.03 | 0.94 |
| 8192 | 32K | 128K | 8.8 | 4,857 | 8,625 | +77.6% | 1.02 | 1.03 |
| 8192 | 32K | 256K | 8.8 | 6,506 | 9,119 | +40.2% | 1.02 | 1.02 |
| 8192 | 32K | 64K | 12 | 3,246 | 7,787 | +139.9% | 1.02 | 1.07 |
| 8192 | 32K | 128K | 12 | 4,854 | 8,582 | +76.8% | 1.02 | 1.04 |
| 8192 | 32K | 256K | 12 | 6,519 | 9,145 | +40.3% | 1.02 | 1.02 |
| 8192 | 32K | 64K | 18 | 3,263 | 8,246 | +152.7% | 1.03 | 1.04 |
| 8192 | 32K | 128K | 18 | 4,868 | 8,733 | +79.4% | 1.03 | 1.03 |
| 8192 | 32K | 256K | 18 | 6,531 | 9,208 | +41.0% | 1.03 | 1.02 |
| 8192 | 16K | 256K | 25 | 6,601 | 9,280 | +40.6% | 1.04 | 1.02 |
| 8192 | 32K | 64K | 25 | 3,264 | 8,229 | +152.1% | 1.03 | 1.04 |
| 8192 | 32K | 128K | 25 | 4,918 | 8,888 | +80.7% | 1.03 | 1.03 |
| 8192 | 8K | 256K | 35 | 6,880 | 9,860 | +43.3% | 1.07 | 1.04 |
| 8192 | 16K | 64K | 50 | 3,373 | 8,915 | +164.3% | 1.02 | 1.02 |
| 8192 | 16K | 128K | 50 | 5,182 | 9,738 | +87.9% | 1.03 | 1.02 |
| 8192 | 32K | 64K | 67.3 | 3,368 | 8,962 | +166.1% | 1.03 | 1.04 |
| 8192 | 32K | 128K | 67.3 | 5,198 | 9,887 | +90.2% | 1.03 | 1.02 |

</details>


## Fidelity across all 168 corrected cells

| | n | within 1% | within 2% | within 5% | worst |
|---|---|---|---|---|---|
| all | 168 | 4 (2.4%) | 43 (25.6%) | 156 (92.9%) | 13% |
| adamw | 84 | 2 (2.4%) | 29 (34.5%) | **84 (100%)** | 4% |
| muon | 84 | 2 (2.4%) | 14 (16.7%) | 72 (85.7%) | 13% |
| all minus muon seq4096 | 147 | 2 (1.4%) | 39 (26.5%) | **146 (99.3%)** | 7% |

### It is a bias, not scatter

All 84 adamw cells sit ABOVE 1.0 — not one cell over-predicted the time.
Divide out the median offset and the picture inverts:

| | median offset | within 1% after removing it | residual sd |
|---|---|---|---|
| adamw (84) | 1.024 | **79.8%** | 0.76% |
| muon minus seq4096 (63) | 1.028 | 77.8% | 0.89% |
| muon seq4096 (21) | 0.943 | 28.6% | **3.70%** |

adamw goes from 2/84 within 1% of the simulator to **67/84 within 1% of a
simulator scaled by 1.024**. The simulator is not noisy; it is consistently
~2.4% optimistic with sub-1% scatter. (Muon seq4096 is not just offset the
wrong way — its scatter is 4x everything else, which is why it reads as a
distinct mechanism rather than the same residual moved.)

### What the sustained-load fix actually fixed

| | median bias | residual sd | structure |
|---|---|---|---|
| original adamw | 1.033 | 1.93% | grew with sequence (1024:1.029 -> 8192:1.046), budget-35 outlier 1.064 |
| corrected adamw | 1.024 | **0.76%** | flat across sequence (1.021-1.028) and budget (1.021-1.029) |

The offset moved a little; the SHAPE moved a lot. The sequence-dependent
component is gone. That is the clock/power settling the 2.5 s floor was meant
to fix. What remains is a uniform multiplicative factor.

### How much of the bias is inter-task dispatch overhead?

About half overall, and essentially ALL of it on task-dense cells.

Measured overhead, from traces of real campaign cell geometries
(`poor_perf/updated_prof/`; NOTE these are the pre-fix revision, so same
geometry under a different plan):

| | per step | % of step | modelled? |
|---|---|---|---|
| inter-step boundary (llama adamw seq8192) | 339.5 ms | 1.45% | mostly (265.7 ms is priced D2H) |
| ...its unmodelled part (client + rtt + dispatch) | 73.8 ms | 0.32% | no |
| **inter-task gaps** (100-1000 us, 2860/step) | **929.2 ms** | **4.0%** | **no** |
| intra-task launch spacing (~6 us, 64k/step) | 388.3 ms | 1.7% | inside the profiled runtime |
| exposed transfers (85.5% memcpy-covered) | 889.2 ms | 3.8% | yes |
| total kernel-idle | 2.39 s | 10.2% | |

Mean inter-task gap: **324.9 us** (adamw trace), 395.0 us (muon trace).

The plans record their own accounting, which leaves almost no room for it. For
`s8192_tr8192_ts262144_b35`: 2210 tasks, compute 24.265 s, makespan 25.052 s —
the schedule is **96.9% compute-saturated**, with 0.787 s (3.1%) of
non-compute. A dispatcher gap per task has nowhere to hide.

Applying the measured gap to every cell's own task count
(`excess = n_tasks x 324.9 us / makespan`):

| tasks/step | cells | mean task ms | predicted excess | actual excess | residual |
|---|---|---|---|---|---|
| 0-400 | 32 | 28.9 | 1.12% | 2.79% | **1.66%** |
| 400-700 | 25 | 21.1 | 1.54% | 2.29% | 0.56% |
| 700-1200 | 19 | 31.0 | 1.05% | 2.20% | 0.68% |
| 1200+ | 8 | 17.0 | 1.91% | 2.70% | **0.05%** |

Median across all 84 adamw cells: predicted 1.15% against an actual 2.44%, so
the measured gap covers **47%** of the bias. Closing it entirely would need
690 us per task, roughly double what the traces show.

Read that gradient carefully: on the task-densest cells dispatch overhead
accounts for the whole gap (residual 0.05%), and the residual grows as cells
get task-lighter. So there are TWO components — a per-task dispatch cost that
dominates when tasks are many, and a roughly constant ~1-1.7% that survives
when they are few. The second is the cache-warm profiling residual the
campaign report already names (the profiler times a task back-to-back on the
same buffers; production runs each block once on freshly-streamed weights).

**A wrong turn worth recording.** The first pass at this tested whether the
bias scales with grad-accum ROUNDS, found r = -0.02, and concluded dispatch
overhead was ruled out. That test was meaningless: task count AND step time
both scale with rounds, so the dispatch FRACTION is invariant to rounds by
construction — r = 0 is exactly what per-task overhead predicts. The variable
that discriminates is task count against makespan, which is the table above.

## Where the unmodelled idle actually is

Magnitudes come from the campaign (`measure` stage); structure comes from
traces of the same cell geometries. Trace WALL times are not used as
measurements — nsys inflates the step (7.271 s traced vs 7.064 s measured on
the muon cell), so traces answer "where" and the campaign answers "how much".

### The cost model itself is sound

| | | |
|---|---|---|
| per-task compute | observed / priced, 400 tasks | **0.0% aggregate** (median 0.993) |
| | block_fwd 0.964, block_bwd 0.997, block_recompute 1.000, optimizer ~1.05 | |
| transfers H2D | priced 52.4 GB/s | observed **54.6 GB/s** |
| transfers D2H | priced 50.8 GB/s | observed **53.6 GB/s** |

Compute is priced accurately in aggregate and transfers are priced
CONSERVATIVELY, so the optimism is not mispricing. It is idle the schedule
does not model.

### Idle by location (muon s1024_tr32K_ts64K_b18, per step)

⚠️ This table is TRACE STRUCTURE from the PRE-FIX revision, which ran a
different plan than the corrected campaign. Use it for WHERE idle occurs and
in what proportions; do NOT compare its absolute numbers against the corrected
plan's schedule.

| where | trace | modelled? |
|---|---|---|
| exposed transfer (>=5 ms, 96.8% memcpy-covered) | 437.2 ms | **yes** — the plan schedules it |
| inter-TASK gaps (100-1000 us, 311/step, mean 395 us) | 122.8 ms | no |
| intra-task launch spacing (<100 us, 10337/step, mean 13.8 us) | 142.8 ms | partly — the profiler's back-to-back bracket contains some |
| inter-STEP boundary, unmodelled part (CLIENT 17.0 + rtt 1.4 + dispatch 1.3) | 19.7 ms | no |
| inter-step boundary, D2H drain | 70.4 ms | yes — priced transfer |
| dead time inside large gaps | 14.5 ms | no |

### What fixing them buys (all cells, campaign magnitudes)

| | median ratio | within 1% | within 2% |
|---|---|---|---|
| adamw as measured | 1.0244 | 2% | 35% |
| adamw, fix inter-task | 1.0114 | 46% | 82% |
| **adamw, fix both** | **1.0085** | **50%** | **90%** |
| muon as measured | 1.0280 | 0% | 16% |
| muon, fix inter-task | 1.0127 | 33% | 67% |
| **muon, fix both** | **1.0099** | **51%** | **71%** |

Decomposition of the median excess:

| | total | inter-task | inter-step | residual |
|---|---|---|---|---|
| adamw | 2.44% | **1.15%** | 0.30% | 0.99% |
| muon | 2.80% | **1.03%** | 0.28% | 1.49% |

So the two known overheads are roughly HALF the gap, inter-task being ~4x
inter-step. Fixing both roughly halves the bias and takes within-2% from
35%/16% to 90%/71%.

### The ~1-1.5% residual: what is established and what is not

ESTABLISHED, entirely from campaign data (pred vs meas of the SAME plan,
147 cells):

- it is not pricing — per-task compute is priced to 0.0% in aggregate and
  transfers are priced CONSERVATIVELY (54.7 GB/s observed vs 50.8 priced);
  both of those are physical measurements and do not depend on which plan ran
- it scales with MEMORY PRESSURE, not with transfer volume:
  r(residual, budget) = -0.42 (adamw) / -0.25 (muon), and by budget it runs
  +1.56/+1.33/+1.55/+0.90/-0.27/-0.30/+0.64% across 8.8..67.3 GiB. At roomy
  budgets dispatch + inter-step account for the whole excess; the residual
  appears only when the budget is tight.

NOT ESTABLISHED. The natural hypothesis is that transfer/compute overlap falls
short of what the schedule assumes, and the simulator demonstrably DOES model
that stall class (its own schedule blames `to_slow:A_*` blocking specific
`block_fwd` tasks). But the only traces available are from the PRE-FIX
revision and therefore ran a DIFFERENT PLAN than the corrected campaign.
Comparing that trace's schedule against this plan's schedule compares two
different schedules, so no magnitude claim about overlap shortfall is
supportable from it. Treat the mechanism as a hypothesis with a budget
correlation behind it, not a measurement.

THE EXPERIMENT THAT WOULD SETTLE IT: capture one nsys trace OF THE CORRECTED
PLAN at a tight budget (8.8-18 GiB, residual +1.5-2.3%) and one at a roomy
budget (50 GiB, residual ~0), then compare each against its own simulated
schedule. Tooling exists (`simsched.py` for the sim side, `stalls.py` /
`lag2.py` for the trace side); it needs H100 time.

Estimates above use the measured mean inter-task gap (324.9 us) and an
inter-step model of 0.28 us per token of the step, which fits both traced
cells (262144 tok -> 73.8 ms, 65536 tok -> 19.7 ms). Both come from pre-fix
traces, so they carry the same caveat: the per-task gap and boundary cost are
dispatcher properties and should transfer across plans, but they are not
measured on the corrected revision.

## Plan-matched traces (2026-08-01): what the dispatcher actually costs

Earlier sections estimated dispatcher cost as `n_tasks x 324.9 us`. **That
model is retracted.** Three cells were re-run executing their OWN saved plans
(`trace_cell.py`, `expect_prog_id` enforced, so trace and plan are the same
program by construction) and traced. What the traces show:

| cell | t_round | ratio | idle before a TASK | idle before a TRANSFER | transfer share |
|---|---|---|---|---|---|
| ts64K b8.8 | 16K | 1.023 | 263.8 ms | **1405.9 ms** | **84%** |
| ts128K b12 | 32K | 1.042 | 168.8 ms | **1384.5 ms** | **89%** |
| ts64K b12 | 32K | 1.043 | 135.5 ms | **813.6 ms** | **86%** |

(GPU-idle from the kernel union, attributed to the item the dispatcher went on
to issue. Four captured steps per cell.)

**Transfer issue is 84-89% of the dispatcher's exposed idle, in every cell.**
Transfers outnumber compute tasks 3-8x in ACTUAL ISSUES -- well above the
plans' 1.52 trigger ratio, because one prefetch_after/offload_after trigger
becomes many memcpys -- at 0.8-1.7x the per-item cost. For the C-core work
this inverts the priority: task dispatch is the minority cost.

### Why the old model was wrong, in three ways

1. **Wrong population.** It counted `len(program.tasks)` and ignored the
   transfer triggers the dispatcher also issues.
2. **Wrong measurement.** 324.9 us came from kernel-union gaps in the
   100-1000 us band, which is a per-ITEM figure dominated by transfers, not a
   per-task cost. A first attempt to measure it from host-side NVTX gaps was
   worse still (8.4 ms median, 16.7 s of "dispatch" in a 30 s capture) --
   those gaps are the dispatcher WAITING on dependencies and pacing, and
   waiting while the GPU works is free.
3. **Wrong arithmetic.** It assumed dispatch idle adds to the prediction. It
   does not: the tr16K cell carries 5.7% of its step in dispatch idle against
   a 2.3% excess, so the plan already anticipates most of it.

### And it does NOT explain the t_round pattern

| cell | dispatch idle / step | as % of step | excess (ratio-1) |
|---|---|---|---|
| tr16K ts64K | 417 ms | 5.7% | 2.3% |
| tr32K ts128K | 388 ms | 2.8% | 4.2% |
| tr32K ts64K | 237 ms | 3.3% | 4.3% |

The cell with the HIGHEST dispatch fraction has the LOWEST excess. So
dispatcher cost is real, large, and overwhelmingly transfer-side -- but it is
not what makes 32K-round cells predict worse than 16K-round cells. That
remains open.

### What still stands

- per-task compute priced to 0.0% in aggregate; transfers priced
  conservatively (54.7 GB/s observed vs 50.8 priced) -- physical measurements
- the excess is real at ~2-4% and reproduces on plan-matched re-runs
  (1.023 / 1.042 / 1.043 here vs 1.014 / 1.033 / 1.031 in the campaign, the
  offset being nsys overhead)
- transfer dispatch is 84-89% of dispatcher idle

### What is retracted

Every projection built on `n_tasks x 324.9 us`: the "inter-task = 1.15% /
1.03%" decomposition, the "fixing inter-task + inter-step reaches 1.0085 /
1.0099" table, and the residual attribution that followed from them. Also the
earlier "memory pressure" reading of the residual (confounded: the paired
within-geometry test shows no budget effect, median -0.20%), and the
0.28 us/token inter-step model (derived from PRE-FIX traces; the step-boundary
fixes cut the client interval ~93%, so it is roughly 4x too high, and on these
6-25 s steps the boundary is 0.05-0.3% regardless).

## NEW: muon over-predicts at seq 4096, and only there

The corrected muon mean ratio (1.008) looks near-perfect, but it is an
average over two populations. Split by sequence length:

| sequence | cells | median ratio | min | cells < 1.00 |
|---|---|---|---|---|
| 1024 | 21 | 1.028 | 1.01 | 0 |
| 2048 | 21 | 1.028 | 1.02 | 0 |
| 4096 | 21 | 0.943 | 0.87 | 21 |
| 8192 | 21 | 1.027 | 1.02 | 0 |

**Every one of the 21 seq-4096 muon cells measures FASTER than predicted**
(median 0.943, worst 0.87 — the simulator says 7.93 s, the engine does it in
6.91 s). The other three sequence lengths sit at ~1.028, indistinguishable
from adamw. And adamw's own seq-4096 cells are normal (median 1.028, min 1.01), so this is not a sequence-length
effect in the planner — it is specific to **muon x seq 4096**.

This is a NEW discrepancy, in the opposite direction from every previously
known one: the simulator has always run slightly optimistic (ratios above 1)
because profiling is cache-warm. Here it is pessimistic by up to 13%, which
means some muon cost at seq 4096 is priced too high. It does not affect the
throughput numbers (those are measured), but any plan chosen for a seq-4096
muon cell was chosen against wrong prices, so the frontier there may not be
the true frontier. Not investigated.

## Caveats

- **The two campaigns are different code revisions.** The corrected run
  carries the production-init profiling fills and the 2.5 s sustained-load
  sampling floor; the original does not. The throughput deltas are the whole
  delta between those revisions, not an isolated measurement of one change,
  and neither revision is recorded in `env.json` — worth adding.
- **adamw lost ~2% and it is systematic** (66 of 69 cells). The planner now
  recomputes more, which is what correctly-priced compute implies, but the
  trade is coming out marginally negative on the real engine. Worth a look
  before treating the corrected frontier as strictly better for adamw.
- **Peak throughput barely moved for adamw** (12,618 -> 12,511 tok/s, -0.8%)
  while muon's peak went 7,657 -> 11,714 (+53%). At 268 effective TFLOP/s the
  original muon cells were running at roughly half of adamw's 512 on identical
  hardware for the same model; they now sit at 507 against adamw's 503. The
  magnitude says the original muon numbers were the anomaly, but this report
  does not identify which change fixed them — that would need a bisect over
  the revision delta, not a comparison of two endpoints.
- 15 of the 84 cells in each campaign have no counterpart in the other (the
  frontier moved), so cell-by-cell figures cover 69.

