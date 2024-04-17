# [DETESTS-Dis IberLEF 2024](https://detests-dis.github.io/corpus/)

## 1. Title

DETEction and classification of racial STereotypes in Spanish - Learning with Disagreement

## 2. Organizers

- Mariona Taulé -- mtaule[at]ub.edu
- Simona Frenda -- simona.frenda[at]unito.it
- Wolfgang Schmeisser-Nieto -- wolfgang.schmeisser[at]ub.edu
- Pol Pastells -- pol.pastells[at]ub.edu
- Alejandro Ariza-Casabona -- alejandro.ariza15[at]ub.edu
- Mireia Farrús -- mfarrus[at]ub.edu
- Paolo Rosso -- prosso[at]dsic.upv.es

## 3. Corpus Description

https://detests-dis.github.io/corpus/

## 4. Number of training instances

9906 (5629 from DETESTS corpus and 4277 from StereoHoax-ES)

## 5. Number of columns

18

## 6. Attribute information

- a) source = {“detests”, “stereohoax”}
- b) id = unique identifier
- c) comment_id = comment identifier
- d) text = sentence or tweet
- e) level1 = previous sentence, refers to “id” (only if source=”detests”)
- f) level2 = previous tweet or comment, refers to “comment_id”
- g) level3 = first tweet or comment, refers to “comment_id”
- h) level4 = news text or racial hoax, refers to “id” column in [“level4.csv” table](https://github.com/clic-ub/DETESTS-Dis/blob/main/level4_table.zip)
- i) stereotype_a1 = individual annotation
- j) stereotype_a2
- k) stereotype_a3
- l) stereotype = majority voting (hard label)
- m) stereotype_soft = softmax normalization (soft label)
- n) implicit_a1
- o) implicit_a2
- p) implicit_a3
- q) implicit
- r) implicit_soft

This set of features will only be available during training. The test dataset will only contain the
following attributes: source, id, comment_id, text, level1, level2, level3, level4.

There are no missing values in this dataset.

## 7. File format

Both [train](https://github.com/clic-ub/DETESTS-Dis/blob/main/training_data.zip) and test files will be provided in CSV format, where each field is comma separated.

## 8. Dataset requirements

a) Please fill in the following registration form to receive the dataset password:
https://forms.gle/CeBdPghgBDi21UaG8
  Once you are registered, we will provide you the password to unzip the files.

b) By participating in this competition, you agree to the following Terms and Conditions:
https://tuit.cat/nZ1eq

c) The training and test sets are available in this repository, in the `data` folder. The password is provided after filling the form in a).

## 9. Baselines

You may reproduce the classical baselines (non-informative, random classifier, TFIDF + SVC, Fast Text + SVC) for tasks 1 and 2 with hard labels by running:

- `py baselines.py` (for the test set)

- `py baselines.py -train data/train_val.csv -test data/validation.csv -folder baselines/validation` (for the validation set, after creating it with the `Examples` notebook)

The baselines using BETO (Cañete et al. 2020) for both tasks with hard and soft labels can be re-created with `beto_baselines.ipynb`.
