# RQ3: Sampling from the LSTM generator (`lstm_long`)

16 samples of 300 characters per row.

## A. Temperature (plain sampling)

| temperature | real words % | distinct-2 % | repetition % | style score | archaic words % | line length |
|---|---|---|---|---|---|---|
| 0.2 | 98.50 | 30.98 | 25.63 | 0.84 | 0.56 | 177.37 |
| 0.5 | 98.35 | 75.17 | 7.47 | 0.77 | 1.55 | 86.29 |
| 0.8 | 92.81 | 92.38 | 3.18 | 0.78 | 3.80 | 71.27 |
| 1.0 | 86.30 | 95.86 | 1.93 | 0.74 | 3.21 | 49.46 |
| 1.3 | 69.67 | 98.31 | 0.59 | 0.66 | 3.08 | 43.70 |
| **real Shakespeare** | **94.74** | **90.48** | **2.21** | **0.75** | **6.62** | **49.78** |

## B. Sampling methods (temperature 0.7)

The spelling filter only lets the model continue a word towards a real word from the training text, so every word is real. Top-p drops the unlikely tail of characters.

| method | real words % | distinct-2 % | repetition % | style score | archaic words % | line length |
|---|---|---|---|---|---|---|
| plain | 94.89 | 87.98 | 3.61 | 0.78 | 3.31 | 70.34 |
| top-p 0.9 | 96.98 | 83.47 | 4.00 | 0.76 | 2.83 | 81.55 |
| spelling filter (app) | 99.70 | 87.91 | 3.40 | 0.73 | 3.10 | 66.31 |
| spelling filter + top-p 0.9 | 99.71 | 82.41 | 4.89 | 0.76 | 3.03 | 91.02 |
| **real Shakespeare** | **94.74** | **90.48** | **2.21** | **0.75** | **6.62** | **49.78** |

## C. Model size (temperature 0.7, spelling filter)

| model | real words % | distinct-2 % | repetition % | style score | archaic words % | line length |
|---|---|---|---|---|---|---|
| lstm_long (256 units) | 99.70 | 87.91 | 3.40 | 0.73 | 3.10 | 66.31 |
| lstm_large (512 units) | 99.30 | 88.75 | 2.92 | 0.71 | 2.21 | 61.29 |
| **real Shakespeare** | **94.74** | **90.48** | **2.21** | **0.75** | **6.62** | **49.78** |

## Examples (B)

### plain
```
My lord, I will be age.
Mark you, Cordelia?
I was in the delight.
Had you so sick and light.
I can not be for a good liege.
Bring it to fallow of contraries the fate indeed thy hand, noble Lord Cassio, in their princes, should be done and both soul he more being no way he hath to hear From heart, and that w
```

### top-p 0.9
```
My lord, I will be a thousand with the lived to this confirmed words, And so she that is not an access my father's children are the born and but one trabiles, And stop me, And not well be death in this instant is to a man of any thousand time when they can not steal my heart with her.
For the bound and the
```

### spelling filter (app)
```
My lord, I will be age.
Mark you, Cordelia?
I was in the delight.
Had you so sick and light.
I can not be for a good liege.
Bring it to fallow of contrariously saw to be must.
That excellent troubled and sparing That I am sorry of no saint of what they can not steal more than the king From her beard, stay f
```

### spelling filter + top-p 0.9
```
My lord, I will be a thousand with the lived to this confirmed words, And so she that is not an access my father's children are the born and but one traitors, that the former you there by the bride To live and the grace should be done and both soul he had lie my father have well.
Oh, that the bound and the
```

## Examples (A)

### T = 0.2
```
My lord, I will not be the man and the lives to the streak the poor and the prince of the state of the brave the world in the state of the man of the married with the best strange men, and the state the contract of the sense and the son of the senate that will be a man to the state of the court of the count
```

### T = 0.5
```
My lord, I will be a thousand with the lived to the sent and desire all the prince, and the way to the breath and mighty man and the speech from the traitors with the beard that the strange dentiel to the man with that which is to see a man that in heart that I think of the white as the world in the filled
```

### T = 0.8
```
My lord, most reques.
If thou weat the lived time is convertance With his popper to a little In all as breathe the cheereling ere I burn a word to charge thee.
But to be Caesar not.
Let's deving her in in part That take so brawl of my accept the garment and fielt, Swear thou hast heard to the bound and the
```

### T = 1.0
```
My lord, more blood at that vow a'mont like a grown suddy, lay working's is pricorelate and way To passed, Worth his contlip here in the plife of ccots of outwoved to be Caesar nature, by the now.
My lord's, in good and know by the last found when they can not stea way here.
He travel, to remain mallasting
```

### T = 1.3
```
My lord, more blood gentle Nvy matter, like a growing, farely thougand'cils prico elace and wances?
Descrevio and my cliving her Octain's bifects.
Antony?
Is while to be Cassius, thes, by teddness!, Well.
You will not, Who bring thou say, Thorss in heart in likete; who hast whet's swife, Let me bear foll-we
```
