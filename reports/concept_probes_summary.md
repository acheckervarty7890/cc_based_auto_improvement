# Concept probes — three generators compared

_Generated 2026-08-23 21:49:13Z._

The same experiment run on three ~50-row synthetic cuts per concept, one per
generating model. Every cell is mean AUROC over that concept's eval splits.

## Mean AUROC by generator and concept (averaged over all four arms)

```
concept    highstakes  hu_ha  instructions
generator                                 
llama70b          NaN  0.838           NaN
llama8b         0.851  0.861          0.67
```

## Mean AUROC by generator, concept and arm

```
concept                       highstakes  hu_ha  instructions
generator config    val_mode                                 
llama70b  seq_ens10 dev              NaN  0.846           NaN
                    split            NaN  0.800           NaN
          single    dev              NaN  0.852           NaN
                    split            NaN  0.855           NaN
llama8b   seq_ens10 dev            0.862  0.866         0.649
                    split          0.841  0.855         0.719
          single    dev            0.874  0.851         0.633
                    split          0.828  0.871         0.678
```

