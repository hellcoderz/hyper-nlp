# hyper-nlp
experiments with hyperdimentional for NLP applications

# idea
## model generation
- create random vectors for each words
- train a model on classification data
- model should contain activation probablities of activation N cells if cell A is activated
    - for example: if cell 2 is activated it might activate cell 4 with prob 0.7 and cell 7 with prob 0.3
- to get these prob P(A->B) = count(A->B)/sum(count(A->N)), where N is each cell activated when cell A is activated

## prediction
- word 1 => 2% active => 10% predicted ...