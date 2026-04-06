# Minimum Release Checklist

Use this checklist before pushing the public repository.

## Content

- [x] Top-level README explains repository scope and limitations
- [x] Code subdirectory README matches the released contents instead of the upstream project layout
- [x] Repository scope accurately reflects that manuscript source is not included
- [x] Evidence directory only contains publication-facing summary files
- [x] Paper-to-evidence mapping file is included

## Cleanliness

- [x] Verify there are no local machine paths in newly generated public files
- [x] Verify there are no cached Python artifacts or notebook checkpoints left in the release tree
- [x] Verify there are no checkpoints, archives, raw logs, or datasets in the release tree

## Reproducibility

- [x] Main experiment entry point is documented
- [x] Retained utility scripts are documented
- [x] Rerun at least one smoke test command after final cleanup

Suggested smoke tests:

```bash
python code/wpheadmixer/run_LTF.py --help
python code/wpheadmixer/scripts/run_repeated_ablation.py --help
python code/wpheadmixer/scripts/interpretability_analysis.py --help
```

## Final Manual Review

- [x] Check that README language matches the paper's cautious claims
- [x] Check that cited evidence files actually exist
- [ ] Check that any reviewer-facing repository or supplement link is anonymized for double-blind review
- [ ] Check that all uploaded public assets comply with the intended publication policy
