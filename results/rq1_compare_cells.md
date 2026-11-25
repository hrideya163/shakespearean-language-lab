# RQ1: RNN vs GRU vs LSTM (character-level generator)

- **bpc** = bits per character (lower is better).
- **memory gain** = bpc on the first 10 characters of a window minus bpc on characters 150-200: how much reading more context helps.
- **gradient N back** = size of the gradient reaching an input N characters in the past, as % of the gradient at the most recent input (vanishing gradients -> near 0).
- `_MATCHED` models have about the same number of recurrent weights as the LSTM.

| model | hidden | recurrent weights (k) | best valid bpc | test bpc | memory gain (bpc) | gradient 10 back (%) | gradient 50 back (%) | steps | train minutes |
|---|---|---|---|---|---|---|---|---|---|
| RNN | 256 | 82.176 | 2.224 | 2.322 | 0.345 | 12.474 | 0.001 | 3000 | 8.020 |
| GRU | 256 | 246.784 | 2.081 | 2.176 | 0.415 | 16.094 | 0.382 | 3000 | 23.193 |
| LSTM | 256 | 328.704 | 2.132 | 2.227 | 0.362 | 9.782 | 0.007 | 3000 | 8.845 |
| RNN_MATCHED | 542 | 328.994 | 2.144 | 2.230 | 0.395 | 22.791 | 0.033 | 3000 | 13.875 |
| GRU_MATCHED | 300 | 328.800 | 2.065 | 2.160 | 0.438 | 16.533 | 0.316 | 3000 | 20.308 |

## Samples (temperature 0.8)

### RNN
```
What way the hours?
No, borted mant and bost thou canst thou walk thy oundering to the tolfor then I ' th ' thousin, and the mole what booch eitter I am for you about your hands I as the best into bear then, to rives him, Was sincuncher.
My lubting again to well enied bloody, Nor eleed or thy flemin
```

### GRU
```
What way the holds?
Coll thee?
Madam, I know thou canst too.
The words, and innemon and your son.
Most grave Kate, and this father can both here authoria.
The power of Brutus is the truth I did thy fury went to ring in the construutious.
What a good father?
Yet seen music is dead!
But what have done
```

### LSTM
```
Bedidy you, gentle?
Come the Clearty, By the devil a grief Marry, speak in the lion and your son.
Most grave me, is man of nothing.
I doon their suits.
There is a corned before our strange master you hearts be merry in the construction that if him these heart, and it is this dead!
But worthy Cold An
```

### RNN_MATCHED
```
By thy yourself as they boy I love the child Whom cause to bear his saly, which I have told you wear on the time, and therefore I will bogs here suins.
There is at my body.
Why, what shall make thy face well to give the fear out out of thee in him to see the eighed Thus to have break it.
Well, let m
```

### GRU_MATCHED
```
What way that common is not o?
Marry, and see how no more that will do, if thou met me to followed on the French is come to the Porpenney.
Where is I alafe her other weary"?
Tis what I am before the priest.
My friend Would a nut is chouth.
Tis his tage?
Well, if I is unquith the fellow or thy flatte
```
