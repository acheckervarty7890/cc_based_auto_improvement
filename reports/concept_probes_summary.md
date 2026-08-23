# Concept probes — three generators compared

_Generated 2026-08-23 23:09:20Z._

The same experiment run on three ~50-row synthetic cuts per concept, one per
generating model. Every cell is mean AUROC over that concept's eval splits.

## Mean AUROC by generator and concept (averaged over all four arms)

```
concept    highstakes  hu_ha  instructions
generator                                 
dsv4pro         0.805  0.887           NaN
llama70b        0.897  0.838         0.797
llama8b         0.851  0.861         0.670
```

## Mean AUROC by generator, concept and arm

```
concept                       highstakes  hu_ha  instructions
generator config    val_mode                                 
dsv4pro   seq_ens10 dev              NaN  0.888           NaN
                    split            NaN  0.898           NaN
          single    dev            0.792  0.888           NaN
                    split          0.818  0.871           NaN
llama70b  seq_ens10 dev            0.918  0.846         0.771
                    split          0.877  0.800         0.813
          single    dev            0.900  0.852         0.778
                    split          0.895  0.855         0.825
llama8b   seq_ens10 dev            0.862  0.866         0.649
                    split          0.841  0.855         0.719
          single    dev            0.874  0.851         0.633
                    split          0.828  0.871         0.678
```

