# Cutting a Zenodo release

This repo is set up for the **GitHub ↔ Zenodo integration**: Zenodo watches the
repository and, whenever you publish a GitHub *Release*, archives that tagged
snapshot and mints a DOI. Metadata for the deposit is read from
[`../.zenodo.json`](../.zenodo.json); a human-readable citation lives in
[`../CITATION.cff`](../CITATION.cff).

## Pre-flight checklist (do this before publishing the release)

Placeholders that MUST be filled or they will appear verbatim on the public DOI record:

- [ ] **`.zenodo.json`** — replace each `"affiliation": "TODO — …"` with the real
      affiliation. Add an `"orcid"` field to each creator, e.g.
      `{"name": "Villacís, David", "affiliation": "…", "orcid": "0000-0002-1825-0097"}`.
      (ORCID is validated by Zenodo — a malformed one is rejected, which is why the
      scaffold omits it rather than shipping a fake.)
- [ ] **`.zenodo.json`** — once the paper DOI exists, add it under
      `related_identifiers`:
      `{"relation": "isSupplementTo", "identifier": "10.xxxx/xxxxx", "scheme": "doi"}`.
- [ ] **`CITATION.cff`** — uncomment/fill the `orcid:` and `affiliation:` lines and the
      `preferred-citation.doi:` once the article DOI is assigned.
- [ ] **`README.md`** — after the first release, paste the Zenodo DOI badge/URL into the
      *Citation* section (the marked TODO).
- [ ] Bump the version consistently if not `1.0.0`: `pyproject.toml` `version`,
      `.zenodo.json` `version`, `CITATION.cff` `version`, and the git tag must all agree.

## One-time setup

1. Sign in at <https://zenodo.org> with your GitHub account (or link them under
   *Account → Linked accounts → GitHub*).
2. Go to <https://zenodo.org/account/settings/github/>, find
   `dvillacis/variational-sparse-hpo`, and flip its toggle **On**. This installs the
   release webhook. (Do this *before* creating the release — Zenodo only archives
   releases created after the toggle is on.)

## Publish the release

```bash
# from a clean main with the metadata committed:
git tag -a v1.0.0 -m "v1.0.0 — Zenodo archival release (COAP companion code)"
git push origin main
git push origin v1.0.0

# create the GitHub Release from the tag (triggers the Zenodo webhook):
gh release create v1.0.0 \
  --title "v1.0.0" \
  --notes "Archival release accompanying the COAP paper. See README and CITATION.cff."
```

Within a minute or so a new deposit appears in your Zenodo GitHub dashboard. Zenodo
mints two DOIs: a **concept DOI** (always resolves to the latest version — cite this in
the paper) and a **version DOI** (this specific release). Add the DOI badge to the README.

## Later versions

Any subsequent GitHub Release (`v1.1.0`, …) is archived automatically under the same
concept DOI. Keep the four version fields (above) in sync with each tag.
