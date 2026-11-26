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
| LSTM_LONG | 256 | 328.704 | 2.069 | 2.166 | 0.416 | 31.217 | 0.116 | 9000 | 46.633 |
| RNN_MATCHED_LONG | 542 | 328.994 | 2.137 | 2.221 | 0.389 | 32.255 | 0.183 | 9000 | 36.447 |
| GRU_MATCHED_LONG | 300 | 328.800 | 2.061 | 2.162 | 0.433 | 30.995 | 2.259 | 9000 | 26.196 |

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

### LSTM_LONG
```
What will you come?
O, let me leave her both as I can hear me, And praise when he hath the last looks to the mountain to the more ready, And either I am for my sabin yet behind, I know not my set you here, if the fortune, name, Nor the thing upon me?
It is to be so Thus to have break there now left
```

### RNN_MATCHED_LONG
```
What's your graces?
O, let me lead the gain To be caunt as mine own sallice confere me to follow you?
Not and burial flust not in repungon their suins.
Tus all office and heart, I ask The world, I ' th ' moint me!
Bring you go.
No.
What have us go to such a weak with us.
If with the minds, and left
```

### GRU_MATCHED_LONG
```
Why, why, thou know'st looked?
Madam, I know thou canst be made her sail, which mortal tyliness and done.
Farewell, my lord.
This reasonows here.
There's some power of the cause I ask him to bed.
I had rather deep so mouth and your churth.
Thus it barely tongue we in unlost, And tell there is the wo
```
