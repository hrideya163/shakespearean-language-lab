# RQ3: Effect of temperature on the LSTM generator

16 samples of 300 characters per temperature.

| temperature | real words % | distinct-2 % | repetition % | style score | archaic words % | line length |
|---|---|---|---|---|---|---|
| 0.2 | 99.18 | 29.35 | 26.00 | 0.82 | 0.36 | 239.80 |
| 0.5 | 96.77 | 73.38 | 7.60 | 0.77 | 1.23 | 128.95 |
| 0.8 | 89.27 | 91.90 | 2.76 | 0.76 | 2.55 | 66.30 |
| 1.0 | 80.45 | 94.59 | 1.17 | 0.71 | 1.81 | 60.84 |
| 1.3 | 64.88 | 99.17 | 0.47 | 0.68 | 1.87 | 43.81 |
| **real Shakespeare** | **94.74** | **90.48** | **2.21** | **0.75** | **6.62** | **49.78** |

## Examples

### T = 0.2
```
My lord, I will be a man and the world and the world of the charge the court of the state of the court of the sent of the world in the world of the streaked the serve the man to the stare of the constant to the beard the subject of my heart to the state and many the more that the seas and the court and the
```

### T = 0.5
```
My lord, I will be a thousand with the like a way in the delice of and court of our a little and the store to the more that Brutus.
Let him for the strangers, And therefore the nature, my lord, Caesar, in the wife of the subtles of my heart to see thee can not strange in the world shall be a better like his
```

### T = 0.8
```
My lord, And there are the prower's but.
Go there is not a lady of and court of one a little In all as breath and my content in the born and blood, In the ocks, and the cause.
That expery the noble To live and the grace should be dounce It hath not hear any likete; who hast what the mistres many is like his
```

### T = 1.0
```
My lord, And there are that the it thus.
Go 't a little vallay portia's pleasure to a little In all as breath and my contlio here is speak if the cctart Juch, And, So be Cassius, these by the now.
I call.
You will not east than our lord it nothing.
But can being them of company to swile, Let me bear followe
```

### T = 1.3
```
My lord, How the quary, Mark you 't but.
Go 't at its, that I have aastrous prico ain loud-calline?
Dut Mevio, Jew cherity.
Breathery macabife of ccotraria, two?
I tourfely payed.
Gest my devin Cherite in panieig, Stand sure-breoth, And Remands with werchip To Did; we is she wed.
Come, to remain, all taken
```
