# RQ5: Meaning drift detection

Benchmark built from 600 test pairs: 1068 candidates (600 clean, 468 drifted).

Meaning guard false alarms on clean translations: **17.2%**

| drift type | examples | guard detection % | AUROC (avg log-prob) | 95% CI | AUROC (min log-prob) | model prefers correct version % |
|---|---|---|---|---|---|---|
| negation | 338 | 91.7 | 0.610 | 0.58 - 0.65 | 0.609 | 90.5 |
| name | 104 | 87.5 | 0.600 | 0.54 - 0.65 | 0.663 | 72.1 |
| number | 26 | 76.9 | 0.816 | 0.75 - 0.87 | 0.836 | 100.0 |

## Where the likelihood was fooled (it preferred the corrupted version)

- **negation**: *Lo , here upon thy cheek the stain doth sit Of an old tear that is not washed off yet .*  
  correct: There's still a stain on your cheek from an old tear that hasn't been washed off yet. (avg log-prob -2.85)  
  corrupted: There's still a stain on your cheek from an old tear that has been washed off yet. (avg log-prob -2.69)
- **negation**: *Pronounce this sentence then : Women may fall when there's no strength in men .*  
  correct: Then repeat this after me: you can't expect women to be faithful when men are so unreliable. (avg log-prob -4.10)  
  corrupted: Then repeat this after me: you can expect women to be faithful when men are so unreliable. (avg log-prob -3.80)
- **negation**: *Without his roe , like a dried herring .*  
  correct: He looks skinny, like a dried herring without its eggs, and he hasn't got his girl. (avg log-prob -3.24)  
  corrupted: He looks skinny, like a dried herring without its eggs, and he has got his girl. (avg log-prob -3.08)
- **name**: *Is Rosaline , whom thou didst love so dear , So soon forsaken ?*  
  correct: Have you given up so quickly on Rosaline, whom you loved so much? (avg log-prob -3.02)  
  corrupted: Have you given up so quickly on Capulets, whom you loved so much? (avg log-prob -3.02)
- **name**: *And is he a man to encounter Tybalt ?*  
  correct: Is he man enough at this point to face off with Tybalt? (avg log-prob -4.13)  
  corrupted: Is he man enough at this point to face off with Capulets? (avg log-prob -4.13)
- **name**: *Nay , I am the very pink of courtesy .*  
  correct: The Right. (avg log-prob -4.23)  
  corrupted: The Rosaline. (avg log-prob -3.79)