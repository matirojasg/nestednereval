# nestednereval
Nested NER eval is a framework that calculates different types of metrics for the nested NER task.

## Installation

To install seqeval, simply run:

```bash
pip install nestednereval
```

## Support features

seqeval supports following schemes:

- IOB2

and following metrics:

| metrics  | description  |
|---|---|
| standard_metric(entities)  | Compute the micro f1-score over all entities.  |
| flat_metric(entities)  | Compute micro f1-score over flat entities.  |
| inner_metric(entities)  | Compute micro f1-score over inner entities.  |
| outer_metric(entities)  | Compute micro f1-score over outer entities.  |
| nested_metric(entities)  | Compute micro f1-score over nested entities.  |
| nesting_metric(entities)  | Compute micro f1-score over nestings.  |

## Usage

Example of usage:

```python
>>> from nestednereval.metrics import standard_metric
>>> from nestednereval.metrics import flat_metric
>>> from nestednereval.metrics import nested_metric
>>> from nestednereval.metrics import nesting_metric
>>> # An example composed of two sentences.
>>> entities = [{"real": [("Body Part", 2, 2), ("Disease", 0, 2)], "pred": [("Body Part", 2, 2), ("Disease", 0, 2)]},
{"real": [("Body Part", 2, 2), ("Disease", 0, 2)], "pred": [("Disease", 0, 2)]},
{"real": [("Medication", 1,1)], "pred": [("Medication", 1,1)]}
]
>>> standard_metric(entities)

>>> flat_metric(entities)

>>> nested_metric(entities)

>>> nesting_metric(entities)

```

## License

[MIT](hhttps://github.com/matirojasg/nested_ner_eval/blob/main/LICENSE)

## Citation

```tex
@misc{seqeval,
  title={{nestednereval}: An extension of Seqeval framework to evaluate task-specific evaluation metrics for nested NER.},
  url={https://github.com/matirojasg/nested_ner_eval},
  note={Software available from https://github.com/matirojasg/nested_ner_eval},
  author={Matías Rojas},
  year={2022},
}
```