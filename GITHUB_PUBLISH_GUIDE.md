# GitHub publication checklist

**Do not advertise this archive as the exact original paper repository** unless original source, weights and datasets have been merged and every item in `REPRODUCIBILITY_CHECKLIST.md` is verified.

## Upload steps

1. Unzip the archive. The extracted root `DPCU-PolSfP-reference/` contains `pyproject.toml` and `README.md`. Open a terminal in that folder.
2. Create a GitHub repository with the correct public/private visibility; if possible reserve the paper-consistent name `DPCU-PolSfP`. Avoid uploading a separate nested folder to the repository root.
3. Before pushing, confirm that no proprietary infrared images, proprietary meshes, personal credentials, or third-party models are present. Confirm all collaborators approve and assign an appropriate license.
4. For a local GitHub clone:

```bash
git init
git add .
git commit -m "Add manuscript-derived DPCU-PolSfP reference implementation"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

5. GitHub Actions should run `python -m pytest -q`. On a separate machine, follow `README.md` and run `scripts/smoke_test.sh` with the *synthetic* scene.
6. Only after replacing the reference implementation with original experimental code, add actual raw data / approved data download locations, exact trained weights, per-condition scripts and verified tables. Large models and datasets may require a separate persistent archive (e.g., Zenodo or an institution-controlled repository). **Never say the datasets are open when they are not.**
7. Add commit hash, release tag and dataset/checkpoint SHA256 into the response to Reviewer #1. Verify public URLs in an incognito browser session.
8. Clearly separate code-derived results from experimental hardware records. In particular, AI-generated apparatus imagery must never be presented as photographs of real experiments.

## Reminder about GitHub's ZIP upload

GitHub's web interface does not automatically unpack a single ZIP into repository source files. Use the **extracted files** (Git CLI is simplest) to make modules browsable and testable; you may additionally attach the ZIP as a GitHub Release asset.
