# Results on 200 test sentences (Shakespeare -> modern)

| System | BLEU | Anchors kept % | Kept-word rate % | Consistency % | ms / sentence (CPU) | Model size MB |
|---|---|---|---|---|---|---|
| Copy input (no change) | 21.6 | 100.0 | 100.0 | - | - | - |
| Dictionary baseline | 23.9 | 100.0 | 99.9 | - | - | - |
| Our LSTM | 23.1 | 82.8 | 91.4 | 100.0 | 58.3 | 25.8 |

LSTM parameters: 6.41 M

## Examples

**Original:** Riddling confession finds but riddling shrift .  
**Human:** A jumbled confession can only receive a jumbled absolution .  
**Baseline:** riddling confession finds but riddling shrift .  
**LSTM:** the devil are finds on the shrift .  

**Original:** Then plainly know my heart's dear love is set On the fair daughter of rich Capulet .  
**Human:** I love rich Capulet's daughter .  
**Baseline:** then plainly know my heart's dear love is set on the fair daughter of rich capulet .  
**LSTM:** then i know my heart's dear love is on the fair daughter of rich .  

**Original:** As mine on hers , so hers is set on mine , And all combined , save what thou must combine By holy marriage .  
**Human:** We're bound to each other in every possible way , except we need you to marry us .  
**Baseline:** as mine on hers , so hers is set on mine , and all combined , save what you must combine by holy marriage .  
**LSTM:** as mine hers , so hers is on mine , and all combined , save what you want to do by .  

**Original:** When and where and how We met , we wooed and made exchange of vow , I'll tell thee as we pass , but this I pray : That thou consent to marry us today .  
**Human:** I'll tell you more later about when and where we met , how we fell in love , and how we exchanged promises , but now I'm begging you : please , agree to marry us today .  
**Baseline:** when and where and how we met , we wooed and made exchange of vow , i'll tell you as we pass , but this i please : that you consent to indeed us today .  
**LSTM:** when we wooed , we wooed and made exchange of vow , i'll tell you as we pass to marry us today .  

**Original:** Holy Saint Francis , what a change is here !  
**Human:** Holy Saint Francis , this is a drastic change !  
**Baseline:** holy saint francis , what a change is here !  
**LSTM:** holy saint francis , what a change here is here !  

**Original:** Is Rosaline , whom thou didst love so dear , So soon forsaken ?  
**Human:** Have you given up so quickly on Rosaline , whom you loved so much ?  
**Baseline:** is rosaline , whom you did love so dear , so soon forsaken ?  
**LSTM:** is rosaline , whom you did love so dear , so soon ?  

**Original:** Young men's love then lies Not truly in their hearts , but in their eyes .  
**Human:** Then young men love with their eyes , not with their hearts .  
**Baseline:** young men's love then lies not truly in their hearts , but in their eyes .  
**LSTM:** young men's love , then don't be truly in their hearts , but in their eyes .  

**Original:** The sun not yet thy sighs from heaven clears , Thy old groans ring yet in my ancient ears .  
**Human:** The groans you used to make are still ringing in my old ears .  
**Baseline:** the sun not yet your sighs from heaven clears , your old groans ring yet in my ancient ears .  
**LSTM:** the sun is yet your sighs , your old groans ring in my ears .  

**Original:** Lo , here upon thy cheek the stain doth sit Of an old tear that is not washed off yet .  
**Human:** There's still a stain on your cheek from an old tear that hasn't been washed off yet .  
**Baseline:** lo , here upon your cheek the stain does sit of an old tear that is not washed off yet .  
**LSTM:** lo , here upon your cheek , the stain of an old tear is not washed yet .  

**Original:** If e'er thou wast thyself and these woes thine , Thou and these woes were all for Rosaline .  
**Human:** If you were ever yourself , and this sadness was yours , you and your sadness were all for Rosaline .  
**Baseline:** if ever you were yourself and these woes yours , you and these woes were all for rosaline .  
**LSTM:** if you wast you , these woes were all these woes , you and these woes is all for rosaline .  
