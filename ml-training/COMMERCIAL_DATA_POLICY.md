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

## Existing Runtime Model

The bundled `Virtual Makeup/faceParsing.mlmodel` is not an approved training
source for the new model. Its origin, source dataset, redistribution terms, and
commercial-use rights must be documented before it can ship in a commercial
build. Remove it from production once the reviewed lip-only model replaces its
runtime role, unless it has a separate reviewed use case.
