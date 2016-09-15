# hyper-nlp
experiments with hyperdimentional for NLP applications

# idea
## model generation
- create random vectors for each words, each domain is a separate vector
- train a model on classification data
- each training sample is appended with its domain at the end so the whole sentence gets classified to that domain
- model should contain activation probablities of activation N cells if cell A is activated
    - for example: if cell 2 is activated it might activate cell 4 with prob 0.7 and cell 7 with prob 0.3
- to get these prob P(A->B) = count(A->B)/sum(count(A->N)), where N is each cell activated when cell A is activated

## prediction
- d = damping factor, which means it will decrease at each prediction after first one
- word 1 => 2% active => 10% predicted (d=1.0) => 20% predicted (d=0.8) => 30%(d=0.5) and so on
- word 2 => 2% active => 10% predicted (d=1.0) => 20% predicted (d=0.8) and so on
- for all words
- at the end filter out top 2% +- few extra and classify the resulting vector into one of the domain