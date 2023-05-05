
# Experiment

All the code below should be executed in the folder of `SketchRegex/DeepSketch`.

We randomly sample 120 examples from the original dataset to run our experiments. This is because OpenAI has prevented us from using its API too frequently --- three request per min. We put the original dataset in `SketchRegex/DeepSketch/datasets/xxx.bak` and keep the sampled dataset in `SketchRegex/DeepSketch/datasets/xxx`. By saying "randomly sample", we actually just keep the first 120 samples in the original dataset and throw out the rest.

For example, the originla Turk dataset is in `SketchRegex/DeepSketch/datasets/Turk.bak` and the sampled dataset is in `SketchRegex/DeepSketch/datasets/Turk`.

## Synthesize candidate regexes from an existing NL2Regex system

### pretrained-MLE & Turk

Use the pretrained-MLE model to synthesize candidate regexes, and check against the Turk dataset.

`python decode.py --dataset Turk --model_id pretrained-MLE --split test`.

`python eval.py --dataset Turk --model_id pretrained-MLE --split test`.

If we want to keep the results before ranking, copy the results to a backup folder. Or else the reranking process will overwrite the results so that the eval.py script can work properly. We made this decision following the open/close principle --- We do not change the original code of the NL2Regex system.

### Rerank the candidate regexes

`python nrranker.py --dataset Turk --model_id pretrained-MLE --split test`.






