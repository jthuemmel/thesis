---
name: refs
description: Acquire, index, and study reference papers. Use when a paper is named in discussion and should be held, when a held paper needs a deeper study, or when a question needs what the corpus already says.
---

# References

`refs/` holds everything: the PDFs, the index at `refs/README.md`, and a study per paper
at `refs/<key>.md`. PDFs are gitignored; the index and the studies are committed.

## Reading the corpus

**Never read `refs/README.md` whole.** It grows to tens of thousands of tokens. Grep it
for the key or the topic and read only the matching section. The same applies to a study:
read the section the question needs.

## Naming

`firstauthor_year`, lowercase, matching the PDF filename and the index entry. For a
preprint the year is the year of the arXiv identifier, not of any later publication. A
same-author, same-year collision takes `a`/`b`/`c` in order of identifier.

## Acquiring

1. Identify the paper before fetching it. Confirm the authors, title, venue, and
   identifier, and report them. A wrong paper indexed confidently is worse than none.
2. Fetch the PDF to `refs/<key>.pdf`. arXiv serves `https://arxiv.org/pdf/<id>`, and
   open-access publishers usually serve a direct PDF.
3. **Verify the download before indexing it.** Check the page count against what the
   source claims and confirm text extracts. Truncated and partial downloads are the
   common failure: the lecture corpus was burned twice, by an 8-of-25-page journal
   download and by a 3-of-21-page truncation.
4. Write the index entry.

## The index entry

Under the relevant heading in `refs/README.md`:

    ### `key.pdf`

    Authors, "Title", venue or arXiv identifier, page count.

    One or two lines: what the paper is, and which concepts it is the reference for.

Pointer level only. Which result supports which claim belongs in the study, not here. The
index is read often and grows without bound, so it stays cheap to read. Add a single line
about which version is held only where the file is not what someone would assume, since
that governs whether the right file is held at all.

## The study

`refs/<key>.md`, beside the PDF, written when a paper turns out to matter. It summarises the paper
and sets out the concepts it is a reference for, methodological and mathematical, with
derivations restated in full so they need not be reconstructed in conversation.

A study is about the paper, not about the question that prompted it. Written once it
stands for any later question, and is extended rather than rewritten when it proves thin.

## When retrieval fails

Old journal papers behind a paywall, and sites serving a bot challenge, cannot be
fetched. Say so, give the full citation and the URL worth trying by hand, and stop. Once
the file is in `refs/`, the index step runs on it normally. The lecture corpus holds a
paper fetched by hand through a browser for exactly this reason, so this is the expected
path, not a failure.
