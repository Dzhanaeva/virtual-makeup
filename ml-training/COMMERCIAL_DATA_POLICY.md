# Commercial Data Policy

## Allowed Inputs

Training data may be used only when all of the following are recorded in the
dataset manifest:

- The dataset has a known owner.
- Commercial ML training use is explicitly approved.
- The project has rights to use the images and annotations.
- Consent or another documented legal basis exists for identifiable faces.
- An internal reviewer and approval date are recorded.

The validator rejects data with missing fields, unknown licenses, research-only
terms, or non-commercial restrictions.

## Excluded Sources

Do not use these sources for the production model without a separate written
license:

- CelebAMask-HQ: its official repository limits use to non-commercial research.
- Weights trained on CelebAMask-HQ, including common face-parsing checkpoints.
- Lips Segmentation Dataset mirrors whose license is unknown.
- Scraped social media, search-engine, or stock-photo images without a training
  license and documented consent basis.

## Public Code

Permissively licensed source code may be reviewed as an architectural reference.
Do not assume that a repository license automatically grants commercial rights
to bundled datasets or pretrained weights. Record those artifacts separately.

## Removed Legacy Model

The legacy `faceParsing.mlmodel` was removed because its origin, source dataset,
redistribution terms, and commercial-use rights were not documented. Do not
reintroduce it or use it as a training source for a new model.
