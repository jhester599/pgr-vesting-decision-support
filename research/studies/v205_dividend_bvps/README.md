**Status: preregistered, not yet executed.** This commit freezes the six
v205 candidates, their features, the Ridge penalty grid, every threshold,
the partition lock and the hand-checked targets before any model is fitted.
No forecast has been scored. The final owner summary replaces this note
after execution.

What is frozen: [preregistration.json](outputs/preregistration.json),
[partition_lock.json](outputs/partition_lock.json) and the label-only
artifacts beside them (targets, features, target audit, annual support,
dividend decomposition and source-vintage audit). All 772 v200 control labels
were recomputed by hand from the pinned DB copy and match v200 within
8.9e-16; the prevailing means match within 2.2e-16. D3 closes under its
preregistered support rule: only three positive annual events exist before
the development boundary.

Two preflight attempts stopped before any fit and are kept in
[verification](outputs/verification/): the first recomputed prevailing means
without v200's one pre-output history label; the second found that the v200
lock pins `research/registry.yaml`, which this study must extend. The
registry is now verified at the v200 execution commit and must keep all 114
pinned entries unchanged; every other consumed file still matches its pin.

The execute phase verifies that this committed preregistration is
byte-identical to what the code rebuilds, and refuses to fit otherwise.
