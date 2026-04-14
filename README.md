# CrowsNest (POC)

Tiny star detection playground.

The vibe:
- generate a synthetic star field
- run centroiding
- spit out clean before/after images
- benchmark + profile when things get spicy

## Quick run

```bash
poetry install
poetry run python crowsnest/poc/run.py --mode run --output-image docs/images/poc_sample.png
```

Use a real image as input:

```bash
poetry run python crowsnest/poc/run.py \
	--mode run \
	--input-source image \
	--input-image /path/to/your/starfield.png \
	--output-image docs/images/poc_real_input.png
```

If needed, force resize to the configured run dimensions:

```bash
poetry run python crowsnest/poc/run.py --mode run --input-source image --input-image /path/to/your/starfield.png --resize-input --width 800 --height 600
```

That command writes two files:
- `docs/images/poc_sample_before.png`
- `docs/images/poc_sample_after.png`

## Sample output

Example log from the current pipeline:

```text
INFO  Detected 19 stars
INFO  Before image saved to: docs/images/poc_sample_before.png
INFO  After image saved to: docs/images/poc_sample_after.png
INFO  Top 5 by flux:
INFO  id=9, x=36.712, y=88.778, flux=13187.00, pixels=57
INFO  id=5, x=191.468, y=69.535, flux=11473.00, pixels=52
INFO  id=15, x=42.964, y=129.296, flux=10717.00, pixels=47
```

### Before

![Before star field](docs/images/poc_sample_before.png)

### After (bounding boxes + centroid markers)

![After star field with detections](docs/images/poc_sample_after.png)

## Benchmark / profiling

```bash
make cprofile
make flamegraph
```

If you want details:

```bash
poetry run python -m crowsnest.poc.eval.profile_centroiding --cprofile-output artifacts/centroiding.prof --profile-detail
```

## Next steps

1. Prove the pipeline stages are functionally correct e2e.
2. Validate centroiding quality with repeatable metrics 
3. Build out the remaining stages through LIS mode and close the loop on attitude solution output.
4. Add acceptance checks for attitude quality
5. Start optimization work aimed at embedded deployment

Short version: first make it right, then make it complete, then make it fast on target hardware.

## Reference

- Zhang, G. (2017). *Star Identification*. https://doi.org/10.1007/978-3-662-53783-1
