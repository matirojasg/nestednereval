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
@inproceedings{rojas-etal-2022-simple,
    title = "Simple Yet Powerful: An Overlooked Architecture for Nested Named Entity Recognition",
    author = "Rojas, Matias  and
      Bravo-Marquez, Felipe  and
      Dunstan, Jocelyn",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.184",
    pages = "2108--2117",
    abstract = "Named Entity Recognition (NER) is an important task in Natural Language Processing that aims to identify text spans belonging to predefined categories. Traditional NER systems ignore nested entities, which are entities contained in other entity mentions. Although several methods have been proposed to address this case, most of them rely on complex task-specific structures and ignore potentially useful baselines for the task. We argue that this creates an overly optimistic impression of their performance. This paper revisits the Multiple LSTM-CRF (MLC) model, a simple, overlooked, yet powerful approach based on training independent sequence labeling models for each entity type. Extensive experiments with three nested NER corpora show that, regardless of the simplicity of this model, its performance is better or at least as well as more sophisticated methods. Furthermore, we show that the MLC architecture achieves state-of-the-art results in the Chilean Waiting List corpus by including pre-trained language models. In addition, we implemented an open-source library that computes task-specific metrics for nested NER. The results suggest that metrics used in previous work do not measure well the ability of a model to detect nested entities, while our metrics provide new evidence on how existing approaches handle the task.",
}
```
