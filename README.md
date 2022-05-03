# nestednereval
Nested NER eval is a framework inspired by seqeval that calculates different types of metrics for the nested NER task.

## Installation

To install nestednereval, simply run:

```bash
pip install git+https://github.com/matirojasg/nestednereval.git
```

## Support features

nestednereval supports following schemes:

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
(1.0, 0.8, 0.888888888888889, 5)
>>> flat_metric(entities)
(1.0, 1.0, 1.0, 1)
>>> nested_metric(entities)
(1.0, 0.75, 0.8571428571428571, 4)
>>> nesting_metric(entities)
(1.0, 0.5, 0.6666666666666666, 2)
```

## License

[MIT](hhttps://github.com/matirojasg/nested_ner_eval/blob/main/LICENSE)

## Citation

```tex
@misc{nestednereval,
  title={{nestednereval}: An extension of Seqeval framework to evaluate task-specific evaluation metrics for nested NER.},
  url={https://github.com/matirojasg/nested_ner_eval},
  note={Software available from https://github.com/matirojasg/nested_ner_eval},
  author={Matías Rojas},
  year={2022},
}
```