# Thesis

## Code style

`utils/einmask.py` is the reference. It was written with care. The rest of `utils/` is
also mine and broadly representative, but this is a multi-year project and those files
carry drift from autocomplete, agents, and haste. Where they disagree with `einmask.py`,
`einmask.py` is right.

**Placement.** `utils/` holds my own importable code and is the style reference, so never
write generated code into it. Generated importable code goes in `analysis/`, the
agent-side equivalent of `utils/`. Call sites run from the repository root because import
routing is not set up, which is why `train.py`, `experiment.py`, and the notebooks live
there. A new file may be generated at root when it is a call site.

Do not take style from `analysis/`, `train.py`, or `experiment.py`. The first two are
agent-generated, and `experiment.py` is mid-refactor into them.

**Type hints.** On parameters, including those with defaults
(`dim: int = 1`, `sigma: float = 0.7`, `horizontal_length: float | list | torch.Tensor`).
Both `Optional[X]` and `X | Y` are in use. Return hints where they clarify, not
everywhere.

**Keyword arguments are spaced: `key = value`.** Both sides, in calls and in defaults.
`dim = network.dim`, `bias = False`, `shape = world.token_shape`. Both `key=value` and
`key= value` are accidental. `einmask.py` still contains fifteen of the half-spaced
form; those are slips, not a second convention.

**Comments are sparse and carry content.** A comment that restates its line is noise.
Write one to name a conceptual group, to record the mathematics, or to warn:

    # I = ½ ln(2π) + ln σ + ½ z^2
    # crps as per Gneiting et al 2005
    # overflows fp16, be careful

A restatement such as `# compute the mean` does not belong. Trivial helpers (`exists`,
`default`, `count_parameters`) carry no comment at all.

Form: short lowercase fragments on their own line above the block they label
(`# sample white noise`, `# gaussian kernels`, `# expand and register`). Not sentences,
not capitalised, no trailing full stop. Trailing inline comments are reserved for a
warning or a citation.

**Never pad whitespace to align comments into columns.** One space before `#`.

**Blank lines mark conceptual groups.** A function or `__init__` reads as a sequence of
phases, each separated by a blank line and named by the comment above it. The
`EinMask.__init__` sequence (config attributes, learnable parameters, I/O,
encoder/decoder, weight initialization) is the pattern to follow.

**Docstrings only where an external reference is needed.** `SphericalDiffusionNoise` has
one because it cites the ECMWF paper it derives from. Most functions have none.

## Method

Do not write custom solvers. Where no analytic formula is available, estimate by Monte
Carlo sampling, reusing the existing sampling code rather than writing a new sampler.

Do not hard-reference specific mask-sampling functions. That code is being reworked and
intermediate results live in `masking.ipynb`, so any reference written now goes stale.
Ask where the current sampler is rather than assuming.

MC runs are cheap here and will not take the local environment down. Run them when an
estimate answers the question on the table, and report the result.

If you think an analytic result exists, show the derivation and let me check it before
any code is written.

## References

`refs/` holds reference PDFs, the index at `refs/README.md`, and a study per paper at
`refs/<key>.md`. Grep the index, never read it whole; it is built to grow past what is
worth loading.

Use the `refs` skill to add a paper. When a paper surfaces in discussion, say what it is
and why it is worth holding, then wait rather than fetching it.

## Working method

State assumptions before implementing, not after. Where a task admits more than one
reading and the readings produce different work, say which one you are taking before
you take it. One sentence, no template.

Stop on contradictions rather than picking a side. This repository holds hand-written
code, agent-generated code, and an unfinished refactor side by side, so conflicting
sources are the normal case. When two places disagree, name both and ask which governs.

Prefer the boring solution and the fewest lines that work. Do not add an abstraction
unless it removes more complexity than it introduces. If a hundred lines would do, do
not write a thousand. `utils/` is compact on purpose.

Correctness before speed. Write the obviously correct version, verify it, then optimise
while preserving behaviour.

After a refactor, list what has become unreachable and ask before deleting it. Do not
delete code you do not understand, and do not remove comments you cannot account for.

## Scope

- Found a pre-existing bug, a performance problem, or behaviour the task did not
  mention? Do not fix, optimise, or extend it. Report it as a follow-up in your summary.
  The exception is when the requested behaviour cannot work without it.
- Ambiguous task? Implement the reading its wording and the surrounding code most
  directly support, and state that assumption. Do not build for the other readings too.
- Verify however you like. Scratch scripts need not be kept, and must not become
  permanent test files.
- Add tests only where I ask, or where this repository already keeps tests for that kind
  of change, sized like the neighbouring files.

This concerns extras only. Implement every behaviour actually asked for, in full.
