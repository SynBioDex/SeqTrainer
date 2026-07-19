# Stage B named-A100 evidence handoff

This directory is the repository copy of the successful Colab capture
`a100-20260719T024912Z`, produced from commit
`68262e4e4c8e6207c208fc5e7e7b31baa2ab191f` on an
`NVIDIA A100-SXM4-40GB`.

The source handoff, including command logs, is retained in Google Drive:

<https://drive.google.com/drive/folders/19IxFycb0Z9xGoz9g2pmUKFsonQH75POA>

The captured JSON, Markdown, and SVG files must not be regenerated on a
different host and presented as this run. The manifest records SHA-256 hashes
for every semantic source JSON. Independently revalidate the copied evidence
with:

```bash
uv run seqtrainer-titans-stage-b-a100-pilot \
  --output-dir artifacts/titans_stage_b/a100 --verify-only
```

The additional `README.md` is repository provenance and is intentionally not
part of the captured manifest checksum set.
