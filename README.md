# Anime AI Art Detect
A BEiT classifier to see if anime art was made by an AI or a human.

View the huggingface [here](https://huggingface.co/saltacc/anime-ai-detect)

### Disclaimer
Like most AI models, this classifier is not 100% accurate. Please do not take the results of this model as fact.

The best version had a 96% accuracy distinguishing aibooru and the images from the imageboard sites. However, the success you have with this model will vary based on the images you are trying to classify.

Here are some biases I have noticed from my testing:

 - Images on aibooru, the site where the AI images were taken from, were high quality AI generations. Low quality AI generations have a higher chance of being misclassified
 - Textual inversions and hypernetworks increase the chance of misclassification

### Training
This model was trained from microsoft/beit-base-patch16-224 for one epoch on 11 thousand images from imageboard sites, and 11 thousand images from aibooru.

You can view the wandb run [here](https://wandb.ai/saltacc/huggingface/runs/2mp30x7j?workspace=user-saltacc).


### Use Case
I don't intend for this model to be more accurate than humans for detecting AI art.
I think the best use cases for this model would be for cases where misclassification isn't a big deal, such as
removing AI art from a training dataset.

# Usage

from URL
```
python3 infer.py --url <url here>
```

from file
```
python3 infer.py --image <file path>
```

or, you can use the [huggingface space](https://huggingface.co/spaces/saltacc/anime-ai-detect)
