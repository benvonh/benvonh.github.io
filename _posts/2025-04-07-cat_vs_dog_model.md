---
title: Cat vs Dog Model
date: 2025-04-07
categories: [AI]
tags: [ai]
---

## Saving a Cats v Dogs Model

This is a minimal example showing how to train a fastai model on Kaggle, and save it so you can use it in your app.


```python
# Make sure we've got the latest version of fastai:
!pip install -Uqq fastai
```

First, import all the stuff we need from fastai:


```python
from fastai.vision.all import *
```

Download and decompress our dataset, which is pictures of dogs and cats:


```python
path = untar_data(URLs.PETS)/'images'
```

We need a way to label our images as dogs or cats. In this dataset, pictures of cats are given a filename that starts with a capital letter:


```python
def is_cat(x): return x[0].isupper() 
```

Now we can create our `DataLoaders`:


```python
dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat,
    item_tfms=Resize(192))
```

... and train our model, a resnet18 (to keep it small and fast):


```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

Now we can export our trained `Learner`. This contains all the information needed to run the model:


```python
learn.export('model.pkl')
```

Finally, open the Kaggle sidebar on the right if it's not already, and find the section marked "Output". Open the `/kaggle/working` folder, and you'll see `model.pkl`. Click on it, then click on the menu on the right that appears, and choose "Download". After a few seconds, your model will be downloaded to your computer, where you can then create your app that uses the model.